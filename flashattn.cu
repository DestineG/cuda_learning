#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

#define CHECK_CUDA(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
} while(0)

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

    // Shared Memory 布局
    extern __shared__ char smem[];
    T* s_q = reinterpret_cast<T*>(smem);
    T* s_k = s_q + Br * D;
    T* s_v = s_k + Bc * D;

    // 搬运 Q 到 Shared Memory
    for (int i = thread_id; i < (Br * D) / 4; i += num_threads) {
        int row = (i * 4) / D;
        int col = (i * 4) % D;
        if (q_block_start + row < q_len) {
            reinterpret_cast<float4*>(s_q + row * D + col)[0] = 
                reinterpret_cast<const float4*>(q_base_ptr + (q_block_start + row) * q_stride_token + col)[0];
        } else {
            reinterpret_cast<float4*>(s_q + row * D + col)[0] = make_float4(0,0,0,0);
        }
    }
    __syncthreads();

    // 初始化 Warp 协作变量
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    int num_warps = num_threads / 32;

    // 每个 Warp 认领一行 Q, 计算过程: (Br, D) @ (Bc, D)^T @ (Bc, D) -> (Br, D)
    for (int q_row = warp_id; q_row < (q_block_end - q_block_start); q_row += num_warps) {
        float row_max = -1e20f;
        float row_sum = 0.0f;
        float acc[D / 32];
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) acc[i] = 0.0f;

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

            // (1, D) @ (1, D)^T @ (1, D) -> (1, D)
            int min_k_row = min(Bc, kv_len - kv_block_start);
            for (int k_row = 0; k_row < min_k_row; ++k_row) {
                
                // 因果掩码判断
                if (is_causal && (q_block_start + q_row < kv_block_start + k_row)) continue;

                // // (1, D) @ (1, D)^T -> (1, 1)
                float attn_score = 0.0f;
                #pragma unroll
                for (int d = lane_id; d < D; d += 32) {
                    attn_score += static_cast<float>(s_q[q_row * D + d]) * static_cast<float>(s_k[k_row * D + d]);
                }
                // Warp 归约
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                    attn_score += __shfl_down_sync(0xffffffff, attn_score, offset);
                attn_score = __shfl_sync(0xffffffff, attn_score, 0);
                attn_score *= scale;

                // Online Softmax
                float old_max = row_max;
                row_max = fmaxf(row_max, attn_score);
                float rescale = expf(old_max - row_max);
                float exp_score = expf(attn_score - row_max);
                
                row_sum = row_sum * rescale + exp_score;

                // (1, 1) @ (1, D) -> (1, D)
                #pragma unroll
                for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                    int d_idx = lane_id + acc_i * 32;
                    acc[acc_i] = acc[acc_i] * rescale + exp_score * static_cast<float>(s_v[k_row * D + d_idx]);
                }
            }
        }

        // 归一化并写回
        #pragma unroll
        for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
            int d_idx = lane_id + acc_i * 32;
            o_base_ptr[(q_block_start + q_row) * o_stride_token + d_idx] = static_cast<T>(acc[acc_i] / row_sum);
        }
    }
}

int main() {
    using T = float;

    // ===== 参数 =====
    const int B = 3;
    const int H = 32;
    const int H_k = 8;
    const int D = 64;

    const int Br = 32;
    const int Bc = 32;

    // ===== 每个序列长度（变长）=====
    int q_lens[B] = {100, 64, 120};
    int k_lens[B] = {120, 80, 100};

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

    int* h_cu_q = (int*)malloc((B + 1) * sizeof(int));
    int* h_cu_k = (int*)malloc((B + 1) * sizeof(int));

    assert(h_Q && h_K && h_V && h_O && h_cu_q && h_cu_k);

    // ===== 初始化数据 =====
    for (int i = 0; i < total_q * H * D; ++i)
        h_Q[i] = rand() / (float)RAND_MAX;

    for (int i = 0; i < total_kv * H_k * D; ++i) {
        h_K[i] = rand() / (float)RAND_MAX;
        h_V[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < total_q * H * D; ++i)
        h_O[i] = 0;

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

    size_t smem_size = (Br + 2 * Bc) * D * sizeof(T);

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

    std::cout << "O[0] = " << h_O[0] << std::endl;

    // ===== free =====
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
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