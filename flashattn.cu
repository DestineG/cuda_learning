#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
} while(0)

template<typename T>
void flashattn_cpu_reference(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    int B,
    int num_heads,
    int num_heads_k,
    int D,
    float scale,
    int q_stride_token,
    int q_stride_head,
    int k_stride_token,
    int k_stride_head,
    int v_stride_token,
    int v_stride_head,
    int o_stride_token,
    int o_stride_head,
    bool is_causal
) {
    for (int b = 0; b < B; ++b) {
        const int q_start = cu_seqlens_q[b];
        const int kv_start = cu_seqlens_k[b];
        const int q_len = cu_seqlens_q[b + 1] - q_start;
        const int kv_len = cu_seqlens_k[b + 1] - kv_start;

        for (int h = 0; h < num_heads; ++h) {
            const int kv_h = h / (num_heads / num_heads_k);

            for (int q_row = 0; q_row < q_len; ++q_row) {
                float row_max = -1e20f;
                float row_sum = 0.0f;
                std::vector<float> acc(D, 0.0f);

                for (int k_row = 0; k_row < kv_len; ++k_row) {
                    if (is_causal && (q_row < k_row)) continue;

                    float score = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const float qv = static_cast<float>(Q[(q_start + q_row) * q_stride_token + h * q_stride_head + d]);
                        const float kv = static_cast<float>(K[(kv_start + k_row) * k_stride_token + kv_h * k_stride_head + d]);
                        score += qv * kv;
                    }
                    score *= scale;

                    const float old_max = row_max;
                    row_max = std::max(row_max, score);
                    const float rescale = std::exp(old_max - row_max);
                    const float exp_score = std::exp(score - row_max);

                    row_sum = row_sum * rescale + exp_score;
                    for (int d = 0; d < D; ++d) {
                        const float vv = static_cast<float>(V[(kv_start + k_row) * v_stride_token + kv_h * v_stride_head + d]);
                        acc[d] = acc[d] * rescale + exp_score * vv;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    O[(q_start + q_row) * o_stride_token + h * o_stride_head + d] = static_cast<T>(acc[d] / row_sum);
                }
            }
        }
    }
}

// Grid 的维度设计为 ((q_len + Br - 1) / Br, num_heads, batch)
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel(
    const T* __restrict__ Q,           // (Total_Q, H, D)
    const T* __restrict__ K,           // (Total_KV, H_k, D)
    const T* __restrict__ V,           // (Total_KV, H_k, D)
    T* __restrict__ O,                  // (Total_Q, H, D)
    const int* __restrict__ cu_seqlens_q, // (B + 1)
    const int* __restrict__ cu_seqlens_k, // (B + 1)
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int num_heads,
    const int num_heads_k,
    const int q_stride_token, const int q_stride_head,
    const int k_stride_token, const int k_stride_head,
    const int v_stride_token, const int v_stride_head,
    const int o_stride_token, const int o_stride_head,
    bool is_causal
) {
    static_assert(D % 32 == 0, "D must be multiple of 32");

    int seq_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    const int q_start = cu_seqlens_q[seq_idx];
    const int kv_start = cu_seqlens_k[seq_idx];
    const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
    const int kv_len = cu_seqlens_k[seq_idx + 1] - kv_start;

    int q_block_start = blockIdx.x * Br;
    if (q_block_start >= q_len) return;
    int q_block_end = min(q_block_start + Br, q_len);

    // 定位当前 Batch 和 Head 的起始指针
    const T* q_base_ptr = Q + q_start * q_stride_token + head_idx * q_stride_head;
    int kv_head_idx = head_idx / (num_heads / num_heads_k);
    const T* k_base_ptr = K + kv_start * k_stride_token + kv_head_idx * k_stride_head;
    const T* v_base_ptr = V + kv_start * v_stride_token + kv_head_idx * v_stride_head;
    T* o_base_ptr = O + q_start * o_stride_token + head_idx * o_stride_head;

    // Shared Memory 布局 —— K/V 块
    extern __shared__ char smem[];
    T* s_k = reinterpret_cast<T*>(smem);
    T* s_v = s_k + Bc * D;

    // 初始化 Warp 协作变量
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    int num_warps = num_threads / 32;
    int q_block_rows = q_block_end - q_block_start;

    // 让整个 block 按 tile 方式推进，每个 tile 里每个 warp 处理一行 Q
    const int warp_handle_row = 4;
    for (int q_tile_start = 0; q_tile_start < q_block_rows; q_tile_start += warp_handle_row * num_warps) {
        int q_row[warp_handle_row];
        #pragma unroll
        for (int i = 0; i < warp_handle_row; ++i) q_row[i] = q_tile_start + warp_id * warp_handle_row + i;
        bool active[warp_handle_row];
        int active_count = 0;
        #pragma unroll
        for (int i = 0; i < warp_handle_row; ++i) {
            active[i] = (q_row[i] < q_block_rows);
            active_count += active[i] ? 1 : 0;
        }

        float row_max[warp_handle_row];
        #pragma unroll
        for (int i = 0; i < active_count; ++i) row_max[i] = -1e20f;
        float row_sum[warp_handle_row];
        #pragma unroll
        for (int i = 0; i < active_count; ++i) row_sum[i] = 0.0f;
        float acc[warp_handle_row][D / 32];
        float q_buf[warp_handle_row][D / 32];

        #pragma unroll
        for (int i = 0; i < active_count; ++i) {
            #pragma unroll
            for (int d = lane_id; d < D; d += 32) {
                q_buf[i][d / 32] = static_cast<float>(q_base_ptr[(q_block_start + q_row[i]) * q_stride_token + d]);
            }
        }

        // 遍历 K/V 块，每次处理 Bc 行 K/V
        // (1, D) @ (Bc, D)^T @ (Bc, D) -> (1, D)
        for (int kv_block_start = 0; kv_block_start < kv_len; kv_block_start += Bc) {

            // 搬运 K/V 到 Shared Memory
            __syncthreads(); // 确保上一轮计算已读完 smem
            for (int i = thread_id; i < (Bc * D) / 4; i += num_threads) {
                int row = (i * 4) / D;
                int col = (i * 4) % D;
                if (kv_block_start + row < kv_len) {
                    reinterpret_cast<float4*>(s_k + row * D + col)[0] = 
                        reinterpret_cast<const float4*>(k_base_ptr + (kv_block_start + row) * k_stride_token + col)[0];
                    reinterpret_cast<float4*>(s_v + row * D + col)[0] = 
                        reinterpret_cast<const float4*>(v_base_ptr + (kv_block_start + row) * v_stride_token + col)[0];
                } else {
                    float4 zero = make_float4(0,0,0,0);
                    reinterpret_cast<float4*>(s_k + row * D + col)[0] = zero;
                    reinterpret_cast<float4*>(s_v + row * D + col)[0] = zero;
                }
            }
            __syncthreads(); // 确保 K/V 搬运完成

            // 如果当前 warp 没有有效的 Q 行，则跳过计算
            if (active_count == 0) {
                continue;
            }

            // (1, D) @ (1, D)^T @ (1, D) -> (1, D)
            int min_k_row = min(Bc, kv_len - kv_block_start);
            for (int k_row = 0; k_row < min_k_row; ++k_row) {
                bool causal_mask[warp_handle_row];
                #pragma unroll
                for (int q_i = 0; q_i < active_count; ++q_i) {
                    int qRow = q_row[q_i];
                    causal_mask[q_i] = is_causal && (q_block_start + qRow < kv_block_start + k_row);
                }

                // (1, D) @ (1, D)^T -> (1, 1)
                float attn_score[warp_handle_row];
                #pragma unroll
                for (int q_i = 0; q_i < active_count; ++q_i) {
                    attn_score[q_i] = 0.0f;
                }
                #pragma unroll
                for (int d = lane_id; d < D; d += 32) {
                    #pragma unroll
                    for (int q_i = 0; q_i < active_count; ++q_i) {
                        if (causal_mask[q_i]) {
                            attn_score[q_i] = -1e20f; // Apply causal mask
                        }
                        else {
                            attn_score[q_i] += static_cast<float>(q_buf[q_i][d / 32]) * static_cast<float>(s_k[k_row * D + d]);
                        }
                    }
                }
                // Warp 归约
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    #pragma unroll
                    for (int q_i = 0; q_i < active_count; ++q_i) {
                        attn_score[q_i] += __shfl_down_sync(0xffffffff, attn_score[q_i], offset);
                    }
                }
                #pragma unroll
                for (int q_i = 0; q_i < active_count; ++q_i) {
                    attn_score[q_i] = __shfl_sync(0xffffffff, attn_score[q_i], 0);
                    attn_score[q_i] *= scale;
                }

                // Online Softmax
                float old_max[warp_handle_row];
                float rescale[warp_handle_row];
                float exp_score[warp_handle_row];
                #pragma unroll
                for (int q_i = 0; q_i < active_count; ++q_i) {
                    old_max[q_i] = row_max[q_i];
                    row_max[q_i] = fmaxf(row_max[q_i], attn_score[q_i]);
                    rescale[q_i] = expf(old_max[q_i] - row_max[q_i]);
                    exp_score[q_i] = expf(attn_score[q_i] - row_max[q_i]);
                    row_sum[q_i] = row_sum[q_i] * rescale[q_i] + exp_score[q_i];
                }

                // (1, 1) @ (1, D) -> (1, D)
                #pragma unroll
                for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                    int d_idx = lane_id + acc_i * 32;
                    #pragma unroll
                    for (int q_i = 0; q_i < active_count; ++q_i) {
                        acc[q_i][acc_i] = acc[q_i][acc_i] * rescale[q_i] + exp_score[q_i] * static_cast<float>(s_v[k_row * D + d_idx]);
                    }
                }
            }
        }

        // 归一化并写回
        #pragma unroll
        for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
            int d_idx = lane_id + acc_i * 32;
            #pragma unroll
            for (int q_i = 0; q_i < active_count; ++q_i) {
                o_base_ptr[(q_block_start + q_row[q_i]) * o_stride_token + d_idx] = static_cast<T>(acc[q_i][acc_i] / row_sum[q_i]);
            }
        }
    }
}

/*
优化思路：
1. 设置 KV 双缓冲(不爆 shared memory)
2. 每个 warp 处理多行 Q(减少最外层 for 循环，增加每个 warp 内的计算量)
*/

int main() {
    using T = float;

    // ===== 参数 =====
    const int B = 16;
    const int H = 32;
    const int H_k = 8;
    const int D = 128;
    const int min_seqlen_q = 256;

    // 4090 shared memory 是 48KB=48*1024 bytes
    // 48*1024 bytes >= (2 * Bc) * D * sizeof(T) = Bc * 2 * 128 * 4 bytes
    // => Bc <= 48*1024 / (2 * 128 * 4) = 48*1024 / 1024 = 48
    const int Br = 32;
    const int Bc = 32;

    // ===== 每个序列长度（变长）=====
    int q_lens[B], k_lens[B];
    for (int i = 0; i < B; ++i) {
        q_lens[i] = min_seqlen_q + (rand() % 128);
        k_lens[i] = q_lens[i] + 1 + (rand() % 32);
    }

    // ===== 计算 total tokens =====
    int total_q = 0;
    int total_kv = 0;
    for (int i = 0; i < B; ++i) {
        total_q += q_lens[i];
        total_kv += k_lens[i];
    }

    // ===== stride（packed layout）=====
    const int q_stride_token = H * D;
    const int q_stride_head  = D;

    const int k_stride_token = H_k * D;
    const int k_stride_head  = D;

    const int v_stride_token = H_k * D;
    const int v_stride_head  = D;

    const int o_stride_token = H * D;
    const int o_stride_head  = D;

    // ===== host malloc =====
    T* h_Q = (T*)malloc(total_q * H * D * sizeof(T));
    T* h_K = (T*)malloc(total_kv * H_k * D * sizeof(T));
    T* h_V = (T*)malloc(total_kv * H_k * D * sizeof(T));
    T* h_O = (T*)malloc(total_q * H * D * sizeof(T));
    T* h_O_ref = (T*)malloc(total_q * H * D * sizeof(T));

    int* h_cu_q = (int*)malloc((B + 1) * sizeof(int));
    int* h_cu_k = (int*)malloc((B + 1) * sizeof(int));

    assert(h_Q && h_K && h_V && h_O && h_O_ref && h_cu_q && h_cu_k);

    srand(0);

    // ===== 初始化数据 =====
    for (int i = 0; i < total_q * H * D; ++i)
        h_Q[i] = rand() / (float)RAND_MAX;

    for (int i = 0; i < total_kv * H_k * D; ++i) {
        h_K[i] = rand() / (float)RAND_MAX;
        h_V[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < total_q * H * D; ++i)
        h_O[i] = 0;

    for (int i = 0; i < total_q * H * D; ++i)
        h_O_ref[i] = 0;

    // ===== 构造 cu_seqlens =====
    h_cu_q[0] = 0;
    h_cu_k[0] = 0;
    for (int i = 0; i < B; ++i) {
        h_cu_q[i + 1] = h_cu_q[i] + q_lens[i];
        h_cu_k[i + 1] = h_cu_k[i] + k_lens[i];
    }

    // ===== 计算最大长度（用于 grid）=====
    int max_seqlen_q = 0;
    int max_seqlen_k = 0;
    for (int i = 0; i < B; ++i) {
        if (q_lens[i] > max_seqlen_q) max_seqlen_q = q_lens[i];
        if (k_lens[i] > max_seqlen_k) max_seqlen_k = k_lens[i];
    }

    // ===== device malloc =====
    T *d_Q, *d_K, *d_V, *d_O;
    int *d_cu_q, *d_cu_k;

    CHECK_CUDA(cudaMalloc(&d_Q, total_q * H * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_K, total_kv * H_k * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_V, total_kv * H_k * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_O, total_q * H * D * sizeof(T)));

    CHECK_CUDA(cudaMalloc(&d_cu_q, (B + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cu_k, (B + 1) * sizeof(int)));

    // ===== copy =====
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, total_q * H * D * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, total_kv * H_k * D * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, total_kv * H_k * D * sizeof(T), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_cu_q, h_cu_q, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cu_k, h_cu_k, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // ===== launch =====
    dim3 grid(
        (max_seqlen_q + Br - 1) / Br,
        H,
        B
    );

    dim3 block(128);

    // only K/V blocks are stored in shared memory now
    size_t smem_size = (2 * Bc) * D * sizeof(T);

    float scale = 1.0f / sqrtf((float)D);

    flashattn_kernel<T, Br, Bc, D><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O,
        d_cu_q, d_cu_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        H,
        H_k,
        q_stride_token, q_stride_head,
        k_stride_token, k_stride_head,
        v_stride_token, v_stride_head,
        o_stride_token, o_stride_head,
        true
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // ===== copy back =====
    CHECK_CUDA(cudaMemcpy(h_O, d_O, total_q * H * D * sizeof(T), cudaMemcpyDeviceToHost));

    flashattn_cpu_reference(
        h_Q, h_K, h_V, h_O_ref,
        h_cu_q, h_cu_k,
        B,
        H,
        H_k,
        D,
        scale,
        q_stride_token, q_stride_head,
        k_stride_token, k_stride_head,
        v_stride_token, v_stride_head,
        o_stride_token, o_stride_head,
        true
    );

    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    float max_abs_err = 0.0f;
    int max_abs_err_idx = -1;
    const int total_out = total_q * H * D;
    for (int i = 0; i < total_out; ++i) {
        const float diff = std::fabs(h_O[i] - h_O_ref[i]);
        sum_abs_err += diff;
        sum_sq_err += static_cast<double>(diff) * static_cast<double>(diff);
        if (diff > max_abs_err) {
            max_abs_err = diff;
            max_abs_err_idx = i;
        }
    }

    int max_b = -1, max_q = -1, max_h = -1, max_d = -1;
    if (max_abs_err_idx >= 0) {
        int token_idx = max_abs_err_idx / (H * D);
        int rem = max_abs_err_idx % (H * D);
        max_h = rem / D;
        max_d = rem % D;
        for (int b = 0; b < B; ++b) {
            if (token_idx >= h_cu_q[b] && token_idx < h_cu_q[b + 1]) {
                max_b = b;
                max_q = token_idx - h_cu_q[b];
                break;
            }
        }
    }

    const double mean_abs_err = sum_abs_err / total_out;
    const double rmse = std::sqrt(sum_sq_err / total_out);
    const float atol = 1e-3f;
    const bool pass = max_abs_err < atol;

    std::cout << "GPU O[0] = " << h_O[0] << std::endl;
    std::cout << "CPU O_ref[0] = " << h_O_ref[0] << std::endl;
    std::cout << "max_abs_err = " << max_abs_err
              << ", mean_abs_err = " << mean_abs_err
              << ", rmse = " << rmse << std::endl;
    if (max_abs_err_idx >= 0) {
        std::cout << "max_abs_err_idx = " << max_abs_err_idx
                  << " (b=" << max_b << ", q=" << max_q
                  << ", h=" << max_h << ", d=" << max_d << ")"
                  << ", GPU = " << h_O[max_abs_err_idx]
                  << ", CPU = " << h_O_ref[max_abs_err_idx]
                  << std::endl;
    }
    std::cout << (pass ? "[PASS] GPU matches CPU reference" : "[FAIL] GPU mismatch vs CPU reference") << std::endl;

    // ===== free =====
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O_ref);
    free(h_cu_q);
    free(h_cu_k);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_cu_q);
    cudaFree(d_cu_k);

    return 0;
}