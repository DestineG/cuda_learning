#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

#define CHECK_CUDA(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
} while(0)

void print_performance(const char* name, float milliseconds, double tflops) {
    std::cout << std::left << std::setw(45) << name
              << std::left << std::setw(10) << " | Time: "
              << std::right << std::setw(12) << std::fixed << std::setprecision(4) << milliseconds
              << std::left << std::setw(5) << " ms"
              << std::left << std::setw(12) << " | TFLOPS: "
              << std::right << std::setw(8) << std::fixed << std::setprecision(4) << tflops
              << std::endl;
}

// Grid 的维度设计为 ((q_len + Br - 1) / Br, num_heads, batch)
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel_v1_base(
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
    for (int q_tile_start = 0; q_tile_start < (q_block_end - q_block_start); q_tile_start += num_warps) {
        int q_row = q_tile_start + warp_id;
        bool active = q_row < (q_block_end - q_block_start);
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
                if (!active) continue;
                
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
        if (active) {
            #pragma unroll
            for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                int d_idx = lane_id + acc_i * 32;
                o_base_ptr[(q_block_start + q_row) * o_stride_token + d_idx] = static_cast<T>(acc[acc_i] / row_sum);
            }
        }
    }
}

// Grid 的维度设计为 ((q_len + Br - 1) / Br, num_heads, batch)
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel_v2_unrolled(
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

    // 让整个 block 按 tile 方式推进，每个 warp 顺序处理两行 Q
    const int q_rows_per_warp = 2;
    for (int q_tile_start = 0; q_tile_start < q_block_rows; q_tile_start += num_warps * q_rows_per_warp) {
        int q_row0 = q_tile_start + warp_id * q_rows_per_warp;
        int q_row1 = q_row0 + 1;
        bool active0 = q_row0 < q_block_rows;
        bool active1 = q_row1 < q_block_rows;

        float row_max0 = -1e20f;
        float row_sum0 = 0.0f;
        float row_max1 = -1e20f;
        float row_sum1 = 0.0f;
        float acc0[D / 32];
        float acc1[D / 32];
        float q_buf0[D / 32];
        float q_buf1[D / 32];

        if (active0) {
            #pragma unroll
            for (int i = 0; i < D / 32; ++i) {
                acc0[i] = 0.0f;
                int d_idx_q = lane_id + i * 32;
                q_buf0[i] = static_cast<float>(q_base_ptr[(q_block_start + q_row0) * q_stride_token + d_idx_q]);
            }
        }

        if (active1) {
            #pragma unroll
            for (int i = 0; i < D / 32; ++i) {
                acc1[i] = 0.0f;
                int d_idx_q = lane_id + i * 32;
                q_buf1[i] = static_cast<float>(q_base_ptr[(q_block_start + q_row1) * q_stride_token + d_idx_q]);
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
            if (!active0 && !active1) {
                continue;
            }

            int min_k_row = min(Bc, kv_len - kv_block_start);
            for (int k_row = 0; k_row < min_k_row; ++k_row) {
                // 第一行 Q
                if (active0) {
                    if (!(is_causal && (q_block_start + q_row0 < kv_block_start + k_row))) {
                        float attn_score0 = 0.0f;
                        #pragma unroll
                        for (int d = lane_id; d < D; d += 32) {
                            attn_score0 += static_cast<float>(q_buf0[d / 32]) * static_cast<float>(s_k[k_row * D + d]);
                        }
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2)
                            attn_score0 += __shfl_down_sync(0xffffffff, attn_score0, offset);
                        attn_score0 = __shfl_sync(0xffffffff, attn_score0, 0);
                        attn_score0 *= scale;

                        float old_max0 = row_max0;
                        row_max0 = fmaxf(row_max0, attn_score0);
                        float rescale0 = expf(old_max0 - row_max0);
                        float exp_score0 = expf(attn_score0 - row_max0);

                        row_sum0 = row_sum0 * rescale0 + exp_score0;

                        #pragma unroll
                        for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                            int d_idx = lane_id + acc_i * 32;
                            acc0[acc_i] = acc0[acc_i] * rescale0 + exp_score0 * static_cast<float>(s_v[k_row * D + d_idx]);
                        }
                    }
                }

                // 第二行 Q
                if (active1) {
                    if (!(is_causal && (q_block_start + q_row1 < kv_block_start + k_row))) {
                        float attn_score1 = 0.0f;
                        #pragma unroll
                        for (int d = lane_id; d < D; d += 32) {
                            attn_score1 += static_cast<float>(q_buf1[d / 32]) * static_cast<float>(s_k[k_row * D + d]);
                        }
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2)
                            attn_score1 += __shfl_down_sync(0xffffffff, attn_score1, offset);
                        attn_score1 = __shfl_sync(0xffffffff, attn_score1, 0);
                        attn_score1 *= scale;

                        float old_max1 = row_max1;
                        row_max1 = fmaxf(row_max1, attn_score1);
                        float rescale1 = expf(old_max1 - row_max1);
                        float exp_score1 = expf(attn_score1 - row_max1);

                        row_sum1 = row_sum1 * rescale1 + exp_score1;

                        #pragma unroll
                        for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                            int d_idx = lane_id + acc_i * 32;
                            acc1[acc_i] = acc1[acc_i] * rescale1 + exp_score1 * static_cast<float>(s_v[k_row * D + d_idx]);
                        }
                    }
                }
            }
        }

        if (active0) {
            #pragma unroll
            for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                int d_idx = lane_id + acc_i * 32;
                o_base_ptr[(q_block_start + q_row0) * o_stride_token + d_idx] = static_cast<T>(acc0[acc_i] / row_sum0);
            }
        }

        if (active1) {
            #pragma unroll
            for (int acc_i = 0; acc_i < D / 32; ++acc_i) {
                int d_idx = lane_id + acc_i * 32;
                o_base_ptr[(q_block_start + q_row1) * o_stride_token + d_idx] = static_cast<T>(acc1[acc_i] / row_sum1);
            }
        }
    }
}

// Grid 的维度设计为 ((q_len + Br - 1) / Br, num_heads, batch)
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel_v3_tiled(
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
    const int warp_handle_row = 2;
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
                    float k_val = static_cast<float>(s_k[k_row * D + d]);
                    #pragma unroll
                    for (int q_i = 0; q_i < active_count; ++q_i) {
                        if (causal_mask[q_i]) {
                            attn_score[q_i] = -1e20f; // Apply causal mask
                        }
                        else {
                            attn_score[q_i] += static_cast<float>(q_buf[q_i][d / 32]) * k_val;
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
                    float v_val = static_cast<float>(s_v[k_row * D + d_idx]);
                    #pragma unroll
                    for (int q_i = 0; q_i < active_count; ++q_i) {
                        acc[q_i][acc_i] = acc[q_i][acc_i] * rescale[q_i] + exp_score[q_i] * v_val;
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

// <<<Grid, Block>>> = <<<(Br_idx, head_idx, seq_idx), (128, 1, 1)>>>
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel_v4_vectorized(
    const T* __restrict__ Q,                    // (Total_Q, H, D)
    const T* __restrict__ K,                    // (Total_KV, H_k, D)
    const T* __restrict__ V,                    // (Total_KV, H_k, D)
    T* __restrict__ O,                          // (Total_Q, H, D)
    const int* __restrict__ cu_seq_len_q,          // (B+1,)
    const int* __restrict__ cu_seq_len_k,          // (B+1,)
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
    /* 静态断言，确保 D 维度是 128-bit 对齐的，以便使用 uint4 进行 vectorized load/store */
    static_assert(D * sizeof(T) % 16 == 0, "D dimension must be 128-bit aligned for vectorization");

    /* 静态断言，确保 D 能够被 Warp Size (32) 整除，以便在 32 个线程间均匀分配寄存器累加器 */
    static_assert(D % 32 == 0, "D must be a multiple of 32 for warp-level tiling");

    /* base_ptr 计算(token_idx*stride_token + head_idx*stride_head) */
    int seq_idx = blockIdx.z;
    int q_start = cu_seq_len_q[seq_idx];
    int q_end = cu_seq_len_q[seq_idx + 1];              // 左闭右开区间 [q_start, q_end)
    int q_block_idx = blockIdx.x;
    int q_token_idx = q_start + q_block_idx * Br;
    if (q_token_idx >= q_end) return;                   // 超出范围的 token 直接返回
    int q_block_end = min(q_token_idx + Br, q_end);
    int q_block_rows = q_block_end - q_token_idx;
    const T* q_base_ptr = (
        Q
        + q_token_idx * q_stride_token
        + blockIdx.y * q_stride_head
    );
    T* o_base_ptr = (
        O
        + q_token_idx * o_stride_token
        + blockIdx.y * o_stride_head
    );
    int kv_token_idx = cu_seq_len_k[seq_idx];
    int kv_token_end = cu_seq_len_k[seq_idx + 1];
    int kv_head_idx = blockIdx.y / (num_heads / num_heads_k);
    const T* k_base_ptr = (
        K
        + kv_token_idx * k_stride_token
        + kv_head_idx * k_stride_head
    );
    const T* v_base_ptr = (
        V
        + kv_token_idx * v_stride_token
        + kv_head_idx * v_stride_head
    );

    /* shared memory(使用 char 作为原始类型，后续通过指针转换为 T*) */
    extern __shared__ char smem_raw[];
    T* s_q = reinterpret_cast<T*>(smem_raw);    // (Br, D)
    T* s_k = s_q + Br * D;                      // (Bc, D)
    T* s_v = s_k + Bc * D;                      // (Bc, D)
    T* s_o = s_q;                               // (Br, D) 结果直接写回 s_q 随后 vectorized store 回全局内存

    /* 将 tileQ(Br, D) 加载到 shared memory(每个 warp 负责一行) */
    constexpr int VEC_SIZE = 16 / sizeof(T);
    constexpr int D_VEC = D / VEC_SIZE;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    for (int q_i = warp_id; q_i < q_block_rows; q_i += num_warps) {
        uint4* s_q_row = reinterpret_cast<uint4*>(&s_q[q_i * D]);
        const uint4* g_q_row = reinterpret_cast<const uint4*>(q_base_ptr + q_i * q_stride_token);

        #pragma unroll
        for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
            s_q_row[d_vec] = g_q_row[d_vec];
        }
    }
    __syncthreads();

    /* 遍历 tileQ(每个 warp 负责一行的 qkv 计算) */
    int q_block_start = q_block_idx * Br;
    for (int q_tile_start = 0; q_tile_start < q_block_rows; q_tile_start += num_warps) {
        int q_row_idx = q_tile_start + warp_id;
        int q_global = q_block_start + q_row_idx;
        float row_max = -1e20f;
        float row_sum = 0.0f;
        float acc[D / 32] = {0.0f};
        float q_buf[D / 32];

        if (q_row_idx < q_block_rows) {
            #pragma unroll
            for (int d_i = lane_id; d_i < D; d_i += 32) {
                q_buf[d_i / 32] = static_cast<float>(s_q[q_row_idx * D + d_i]);
            }
        }

        /* 遍历 tileK & tileV */
        for (int kv_tile_start = 0; kv_tile_start < kv_token_end - kv_token_idx; kv_tile_start += Bc) {
            int kv_tile_global_start = kv_tile_start;
            if (is_causal && kv_tile_global_start >= q_block_end) break;            // 如果是 causal 模式且整个 tileK 都在 tileQ 的右边，则后续的 tileK 都不需要处理了，直接 break

            // 将 tileK & tileV(Br, D) 加载到 shared memory(每个 warp 负责一行, 所有线程都参与搬运)
            int tileKV_rows = min(Bc, kv_token_end - kv_token_idx - kv_tile_start);
            for (int k_i = warp_id; k_i < tileKV_rows; k_i += num_warps) {
                uint4* s_k_row = reinterpret_cast<uint4*>(&s_k[k_i * D]);
                uint4* s_v_row = reinterpret_cast<uint4*>(&s_v[k_i * D]);
                const uint4* g_k_row = reinterpret_cast<const uint4*>(k_base_ptr + (kv_tile_start + k_i) * k_stride_token);
                const uint4* g_v_row = reinterpret_cast<const uint4*>(v_base_ptr + (kv_tile_start + k_i) * v_stride_token);

                #pragma unroll
                for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
                    s_k_row[d_vec] = g_k_row[d_vec];
                    s_v_row[d_vec] = g_v_row[d_vec];
                }
            }
            __syncthreads();

            /* 核心计算逻辑 */
            // 分配到 q_row 的 warp 参与接下来的计算
            if (q_row_idx < q_block_rows) {
                for (int k_i = 0; k_i < tileKV_rows; ++k_i) {
                    float attn_score = 0.0f;
                    if (!(is_causal && kv_tile_global_start + k_i > q_global)) {
                        #pragma unroll                                              // 每个线程计算部分点积
                        for (int d_i = 0; d_i < D / 32; ++d_i) {
                            float k_val = static_cast<float>(s_k[k_i * D + d_i * 32 + lane_id]);
                            attn_score += q_buf[d_i] * k_val;
                        }
                        #pragma unroll                                              // warp 内规约
                        for (int offset = 16; offset > 0; offset /= 2) {
                            attn_score += __shfl_down_sync(0xffffffff, attn_score, offset);
                        }
                        attn_score = __shfl_sync(0xffffffff, attn_score, 0);        // 每个线程都拿到最终的 attn_score
                        attn_score *= scale;                                        // 应用缩放
                    } else {
                        attn_score = -1e20f;                             // 整个 token 都被 mask 掉
                    }

                    float old_row_max = row_max;                                // online softmax
                    row_max = fmaxf(row_max, attn_score);
                    float rescale = expf(old_row_max - row_max);
                    float exp_score = expf(attn_score - row_max);
                    row_sum = row_sum * rescale + exp_score;

                    #pragma unroll
                    for (int d_i = 0; d_i < D / 32; ++d_i) {
                        float v_val = static_cast<float>(s_v[k_i * D + d_i * 32 + lane_id]);
                        acc[d_i] = acc[d_i] * rescale + exp_score * v_val;
                    }
                }
            }
            __syncthreads();
        }

        // 将 acc 写回 shared 中 q_row 的位置, 最后整个结果计算完毕用 vectorized store 写回全局内存
        if (q_row_idx < q_block_rows) {
            #pragma unroll
            for (int d_i = lane_id; d_i < D; d_i += 32) {
                float inv_sum = row_sum > 0 ? 1.f / row_sum : 0.f;
                s_o[q_row_idx * D + d_i] = static_cast<T>(acc[d_i / 32] * inv_sum);
            }
        }
    }
    __syncthreads();

    /* 将结果从 shared memory 写回全局内存(每个 warp 负责一行) */
    for (int q_i = warp_id; q_i < q_block_rows; q_i += num_warps) {
        uint4* s_o_row = reinterpret_cast<uint4*>(&s_o[q_i * D]);
        uint4* g_o_row = reinterpret_cast<uint4*>(o_base_ptr + q_i * o_stride_token);

        #pragma unroll
        for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
            g_o_row[d_vec] = s_o_row[d_vec];
        }
    }
}


// <<<Grid, Block>>> = <<<(Br_idx, head_idx, seq_idx), (128, 1, 1)>>>
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel_v5_async_vectorized(
    const T* __restrict__ Q,                    // (Total_Q, H, D)
    const T* __restrict__ K,                    // (Total_KV, H_k, D)
    const T* __restrict__ V,                    // (Total_KV, H_k, D)
    T* __restrict__ O,                          // (Total_Q, H, D)
    const int* __restrict__ cu_seq_len_q,          // (B+1,)
    const int* __restrict__ cu_seq_len_k,          // (B+1,)
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
    /* 静态断言，确保 D 维度是 128-bit 对齐的，以便使用 uint4 进行 vectorized load/store */
    static_assert(D * sizeof(T) % 16 == 0, "D dimension must be 128-bit aligned for vectorization");

    /* 静态断言，确保 D 能够被 Warp Size (32) 整除，以便在 32 个线程间均匀分配寄存器累加器 */
    static_assert(D % 32 == 0, "D must be a multiple of 32 for warp-level tiling");

    /* base_ptr 计算(token_idx*stride_token + head_idx*stride_head) */
    int seq_idx = blockIdx.z;
    int q_start = cu_seq_len_q[seq_idx];
    int q_end = cu_seq_len_q[seq_idx + 1];              // 左闭右开区间 [q_start, q_end)
    int q_block_idx = blockIdx.x;
    int q_token_idx = q_start + q_block_idx * Br;
    if (q_token_idx >= q_end) return;                   // 超出范围的 token 直接返回
    int q_block_end = min(q_token_idx + Br, q_end);
    int q_block_rows = q_block_end - q_token_idx;
    const T* q_base_ptr = (
        Q
        + q_token_idx * q_stride_token
        + blockIdx.y * q_stride_head
    );
    T* o_base_ptr = (
        O
        + q_token_idx * o_stride_token
        + blockIdx.y * o_stride_head
    );
    int kv_token_idx = cu_seq_len_k[seq_idx];
    int kv_token_end = cu_seq_len_k[seq_idx + 1];
    int kv_head_idx = blockIdx.y / (num_heads / num_heads_k);
    const T* k_base_ptr = (
        K
        + kv_token_idx * k_stride_token
        + kv_head_idx * k_stride_head
    );
    const T* v_base_ptr = (
        V
        + kv_token_idx * v_stride_token
        + kv_head_idx * v_stride_head
    );

    /* shared memory(使用 char 作为原始类型，后续通过指针转换为 T*) */
    extern __shared__ char smem_raw[];
    T* s_q = reinterpret_cast<T*>(smem_raw);            // (Br, D)
    T* s_k = s_q + Br * D;                              // (2, Bc, D)
    T* s_v = s_k + 2 * Bc * D;                          // (2, Bc, D)
    T* s_o = s_q;                                       // (Br, D) 结果直接写回 s_q 随后 vectorized store 回全局内存
    bool write_ptr_toggle = 0;                          // 双缓冲切换标志位
    bool read_ptr_toggle = 1 - write_ptr_toggle;        // 双缓冲切换标志位


    /* 将 tileQ(Br, D) 加载到 shared memory(每个 warp 负责一行) */
    constexpr int VEC_SIZE = 16 / sizeof(T);
    constexpr int D_VEC = D / VEC_SIZE;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    // 预读取 tileKV
    int first_tile_rows = min(Bc, kv_token_end - kv_token_idx);
    for (int k_i = warp_id; k_i < first_tile_rows; k_i += num_warps) {
        #pragma unroll
        for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
            __pipeline_memcpy_async(
                &s_k[write_ptr_toggle * Bc * D + k_i * D + d_vec * VEC_SIZE], 
                k_base_ptr + k_i * k_stride_token + d_vec * VEC_SIZE, 
                16
            );
            __pipeline_memcpy_async(
                &s_v[write_ptr_toggle * Bc * D + k_i * D + d_vec * VEC_SIZE], 
                v_base_ptr + k_i * v_stride_token + d_vec * VEC_SIZE, 
                16
            );
        }
    }
    __pipeline_commit();

    for (int q_i = warp_id; q_i < q_block_rows; q_i += num_warps) {
        uint4* s_q_row = reinterpret_cast<uint4*>(&s_q[q_i * D]);
        const uint4* g_q_row = reinterpret_cast<const uint4*>(q_base_ptr + q_i * q_stride_token);

        #pragma unroll
        for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
            s_q_row[d_vec] = g_q_row[d_vec];
        }
    }
    __syncthreads();

    /* 遍历 tileQ(每个 warp 负责一行的 qkv 计算) */
    int q_block_start = q_block_idx * Br;
    for (int q_tile_start = 0; q_tile_start < q_block_rows; q_tile_start += num_warps) {
        int q_row_idx = q_tile_start + warp_id;
        int q_global = q_block_start + q_row_idx;
        float row_max = -1e20f;
        float row_sum = 0.0f;
        float acc[D / 32] = {0.0f};
        float q_buf[D / 32];

        if (q_row_idx < q_block_rows) {
            #pragma unroll
            for (int d_i = lane_id; d_i < D; d_i += 32) {
                q_buf[d_i / 32] = static_cast<float>(s_q[q_row_idx * D + d_i]);
            }
        }

        /* 遍历 tileK & tileV */
        for (int kv_tile_start = 0; kv_tile_start < kv_token_end - kv_token_idx; kv_tile_start += Bc) {
            int kv_tile_global_start = kv_tile_start;
            if (is_causal && kv_tile_global_start >= q_block_end) break;            // 如果是 causal 模式且整个 tileK 都在 tileQ 的右边，则后续的 tileK 都不需要处理了，直接 break

            int current_tile_rows = min(Bc, kv_token_end - kv_token_idx - kv_tile_start);

            // 将 tileK & tileV(Br, D) 加载到 shared memory(每个 warp 负责一行, 所有线程都参与搬运)
            __pipeline_wait_prior(0);
            __syncthreads();
            read_ptr_toggle = write_ptr_toggle;             // 切换读指针
            write_ptr_toggle = 1 - write_ptr_toggle;        // 切换写指针
            int next_kv_tile_start = kv_tile_start + Bc;
            if (next_kv_tile_start < kv_token_end - kv_token_idx) {
                int next_tile_rows = min(Bc, kv_token_end - kv_token_idx - next_kv_tile_start);
                for (int k_i = warp_id; k_i < next_tile_rows; k_i += num_warps) {
                    #pragma unroll
                    for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
                        __pipeline_memcpy_async(
                            &s_k[write_ptr_toggle * Bc * D + k_i * D + d_vec * VEC_SIZE], 
                            k_base_ptr + (next_kv_tile_start + k_i) * k_stride_token + d_vec * VEC_SIZE, 
                            16
                        );
                        __pipeline_memcpy_async(
                            &s_v[write_ptr_toggle * Bc * D + k_i * D + d_vec * VEC_SIZE], 
                            v_base_ptr + (next_kv_tile_start + k_i) * v_stride_token + d_vec * VEC_SIZE, 
                            16
                        );
                    }
                }
                __pipeline_commit();
            }

            /* 核心计算逻辑 */
            // 分配到 q_row 的 warp 参与接下来的计算
            if (q_row_idx < q_block_rows) {
                for (int k_i = 0; k_i < current_tile_rows; ++k_i) {
                    float attn_score = 0.0f;
                    if (!(is_causal && kv_tile_global_start + k_i > q_global)) {
                        #pragma unroll                                              // 每个线程计算部分点积
                        for (int d_i = 0; d_i < D / 32; ++d_i) {
                            float k_val = static_cast<float>(s_k[read_ptr_toggle * Bc * D + k_i * D + d_i * 32 + lane_id]);
                            attn_score += q_buf[d_i] * k_val;
                        }
                        #pragma unroll                                              // warp 内规约
                        for (int offset = 16; offset > 0; offset /= 2) {
                            attn_score += __shfl_down_sync(0xffffffff, attn_score, offset);
                        }
                        attn_score = __shfl_sync(0xffffffff, attn_score, 0);        // 每个线程都拿到最终的 attn_score
                        attn_score *= scale;                                        // 应用缩放
                    } else {
                        attn_score = -1e20f;                             // 整个 token 都被 mask 掉
                    }

                    float old_row_max = row_max;                                // online softmax
                    row_max = fmaxf(row_max, attn_score);
                    float rescale = expf(old_row_max - row_max);
                    float exp_score = expf(attn_score - row_max);
                    row_sum = row_sum * rescale + exp_score;

                    #pragma unroll
                    for (int d_i = 0; d_i < D / 32; ++d_i) {
                        float v_val = static_cast<float>(s_v[read_ptr_toggle * Bc * D + k_i * D + d_i * 32 + lane_id]);
                        acc[d_i] = acc[d_i] * rescale + exp_score * v_val;
                    }
                }
            }
        }
        /* 计算最后一个 tileKV */

        // 将 acc 写回 shared 中 q_row 的位置, 最后整个结果计算完毕用 vectorized store 写回全局内存
        if (q_row_idx < q_block_rows) {
            #pragma unroll
            for (int d_i = lane_id; d_i < D; d_i += 32) {
                float inv_sum = row_sum > 0 ? 1.f / row_sum : 0.f;
                s_o[q_row_idx * D + d_i] = static_cast<T>(acc[d_i / 32] * inv_sum);
            }
        }
    }
    __syncthreads();

    /* 将结果从 shared memory 写回全局内存(每个 warp 负责一行) */
    for (int q_i = warp_id; q_i < q_block_rows; q_i += num_warps) {
        uint4* s_o_row = reinterpret_cast<uint4*>(&s_o[q_i * D]);
        uint4* g_o_row = reinterpret_cast<uint4*>(o_base_ptr + q_i * o_stride_token);

        #pragma unroll
        for (int d_vec = lane_id; d_vec < D_VEC; d_vec += 32) {
            g_o_row[d_vec] = s_o_row[d_vec];
        }
    }
}


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

template<typename T>
void test_cpu_reference(
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
    bool is_causal,
    const char* name
) {
    auto start = std::chrono::high_resolution_clock::now();
    flashattn_cpu_reference(
        Q, K, V, O,
        cu_seqlens_q, cu_seqlens_k,
        B,
        num_heads,
        num_heads_k,
        D,
        scale,
        q_stride_token, q_stride_head,
        k_stride_token, k_stride_head,
        v_stride_token, v_stride_head,
        o_stride_token, o_stride_head,
        is_causal
    );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    const double ops = 2.0 * B * num_heads * D;
    const double tflops = (ops / (duration.count() / 1000.0)) / 1e12;
    print_performance(name, duration.count(), tflops);
}

template<typename T>
bool verify_attention_output(
    const T* gpu_out,
    const T* cpu_ref,
    int total_out,
    const char* kernel_name
) {
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    float max_abs_err = 0.0f;
    int max_abs_err_idx = -1;

    for (int i = 0; i < total_out; ++i) {
        const float diff = std::fabs(static_cast<float>(gpu_out[i] - cpu_ref[i]));
        sum_abs_err += diff;
        sum_sq_err += static_cast<double>(diff) * static_cast<double>(diff);
        if (diff > max_abs_err) {
            max_abs_err = diff;
            max_abs_err_idx = i;
        }
    }

    const double mean_abs_err = sum_abs_err / total_out;
    const double rmse = std::sqrt(sum_sq_err / total_out);
    const bool pass = max_abs_err < 1e-3f;

    std::cout << kernel_name << " max_abs_err = " << max_abs_err
              << ", mean_abs_err = " << mean_abs_err
              << ", rmse = " << rmse << std::endl;
    if (max_abs_err_idx >= 0) {
        std::cout << kernel_name << " max_abs_err_idx = " << max_abs_err_idx
                  << ", GPU = " << gpu_out[max_abs_err_idx]
                  << ", CPU = " << cpu_ref[max_abs_err_idx]
                  << std::endl;
    }
    std::cout << (pass ? "[PASS] " : "[FAIL] ") << kernel_name << " matches CPU reference" << std::endl;

    return pass;
}

template<typename T, int Br, int Bc, int D, typename KernelLauncher>
void test_attention_kernel(
    const T* d_Q,
    const T* d_K,
    const T* d_V,
    T* d_O,
    const int* d_cu_q,
    const int* d_cu_k,
    int B,
    int max_seqlen_q,
    int max_seqlen_k,
    float scale,
    int num_heads,
    int num_heads_k,
    int q_stride_token, int q_stride_head,
    int k_stride_token, int k_stride_head,
    int v_stride_token, int v_stride_head,
    int o_stride_token, int o_stride_head,
    bool is_causal,
    int total_out,
    const T* h_cpu_ref,
    T* h_gpu_out,
    const char* name,
    KernelLauncher launcher,
    size_t smem_size
) {
    dim3 grid((max_seqlen_q + Br - 1) / Br, num_heads, B);
    dim3 block(128);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    launcher(grid, block, smem_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    const double ops = 2.0 * total_out * D;
    const double tflops = (ops / (milliseconds / 1000.0)) / 1e12;
    print_performance(name, milliseconds, tflops);

    CHECK_CUDA(cudaMemcpy(h_gpu_out, d_O, total_out * sizeof(T), cudaMemcpyDeviceToHost));
    verify_attention_output(h_gpu_out, h_cpu_ref, total_out, name);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    using T = float;

    const int B = 8;
    const int H = 32;
    const int H_k = 8;
    const int D = 64;
    const int Br = 32;
    const int Bc = 32;
    const int min_seqlen_q = 512;

    int q_lens[B], k_lens[B];
    for (int i = 0; i < B; ++i) {
        q_lens[i] = min_seqlen_q + (rand() % 64);
        k_lens[i] = q_lens[i] + 1 + (rand() % 32);
    }

    int total_q = 0;
    int total_kv = 0;
    for (int i = 0; i < B; ++i) {
        total_q += q_lens[i];
        total_kv += k_lens[i];
    }

    const int q_stride_token = H * D;
    const int q_stride_head = D;
    const int k_stride_token = H_k * D;
    const int k_stride_head = D;
    const int v_stride_token = H_k * D;
    const int v_stride_head = D;
    const int o_stride_token = H * D;
    const int o_stride_head = D;

    T* h_Q = (T*)malloc(total_q * H * D * sizeof(T));
    T* h_K = (T*)malloc(total_kv * H_k * D * sizeof(T));
    T* h_V = (T*)malloc(total_kv * H_k * D * sizeof(T));
    T* h_O = (T*)malloc(total_q * H * D * sizeof(T));
    T* h_O_ref = (T*)malloc(total_q * H * D * sizeof(T));
    T* h_O_tmp = (T*)malloc(total_q * H * D * sizeof(T));
    int* h_cu_q = (int*)malloc((B + 1) * sizeof(int));
    int* h_cu_k = (int*)malloc((B + 1) * sizeof(int));

    assert(h_Q && h_K && h_V && h_O && h_O_ref && h_O_tmp && h_cu_q && h_cu_k);

    srand(0);
    for (int i = 0; i < total_q * H * D; ++i) {
        h_Q[i] = rand() / (float)RAND_MAX;
        h_O[i] = 0;
        h_O_ref[i] = 0;
        h_O_tmp[i] = 0;
    }
    for (int i = 0; i < total_kv * H_k * D; ++i) {
        h_K[i] = rand() / (float)RAND_MAX;
        h_V[i] = rand() / (float)RAND_MAX;
    }

    h_cu_q[0] = 0;
    h_cu_k[0] = 0;
    for (int i = 0; i < B; ++i) {
        h_cu_q[i + 1] = h_cu_q[i] + q_lens[i];
        h_cu_k[i + 1] = h_cu_k[i] + k_lens[i];
    }

    int max_seqlen_q = 0;
    int max_seqlen_k = 0;
    for (int i = 0; i < B; ++i) {
        max_seqlen_q = std::max(max_seqlen_q, q_lens[i]);
        max_seqlen_k = std::max(max_seqlen_k, k_lens[i]);
    }

    T *d_Q, *d_K, *d_V, *d_O;
    int *d_cu_q, *d_cu_k;
    CHECK_CUDA(cudaMalloc(&d_Q, total_q * H * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_K, total_kv * H_k * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_V, total_kv * H_k * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_O, total_q * H * D * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_cu_q, (B + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cu_k, (B + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, total_q * H * D * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, total_kv * H_k * D * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, total_kv * H_k * D * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cu_q, h_cu_q, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cu_k, h_cu_k, (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

    const float scale = 1.0f / sqrtf((float)D);
    const bool is_causal = true;
    const int total_out = total_q * H * D;

    std::cout << "Matrix Size: B=" << B
              << ", H=" << H
              << ", H_k=" << H_k
              << ", D=" << D << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    auto run_gpu_kernel = [&](const char* name, auto launcher, size_t smem_size) {
        std::cout << std::string(85, '-') << std::endl;
        test_attention_kernel<T, Br, Bc, D>(
            d_Q, d_K, d_V, d_O,
            d_cu_q, d_cu_k,
            B,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            H,
            H_k,
            q_stride_token, q_stride_head,
            k_stride_token, k_stride_head,
            v_stride_token, v_stride_head,
            o_stride_token, o_stride_head,
            is_causal,
            total_out,
            h_O_ref,
            h_O_tmp,
            name,
            launcher,
            smem_size
        );
    };

    test_cpu_reference(
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
        is_causal,
        "CPU FlashAttention Reference"
    );

    run_gpu_kernel(
        "GPU flashattn_kernel_v1_base",
        [&](dim3 grid, dim3 block, size_t smem_size) {
            flashattn_kernel_v1_base<T, Br, Bc, D><<<grid, block, smem_size>>>(
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
                is_causal
            );
        },
        (Br + 2 * Bc) * D * sizeof(T)
    );

    run_gpu_kernel(
        "GPU flashattn_kernel_v2_unrolled",
        [&](dim3 grid, dim3 block, size_t smem_size) {
            flashattn_kernel_v2_unrolled<T, Br, Bc, D><<<grid, block, smem_size>>>(
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
                is_causal
            );
        },
        (2 * Bc) * D * sizeof(T)
    );

    run_gpu_kernel(
        "GPU flashattn_kernel_v3_tiled",
        [&](dim3 grid, dim3 block, size_t smem_size) {
            flashattn_kernel_v3_tiled<T, Br, Bc, D><<<grid, block, smem_size>>>(
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
                is_causal
            );
        },
        (2 * Bc) * D * sizeof(T)
    );

    run_gpu_kernel(
        "GPU flashattn_kernel_v4_vectorized",
        [&](dim3 grid, dim3 block, size_t smem_size) {
            flashattn_kernel_v4_vectorized<T, Br, Bc, D><<<grid, block, smem_size>>>(
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
                is_causal
            );
        },
        (Br + 2 * Bc) * D * sizeof(T)
    );

    run_gpu_kernel(
        "GPU flashattn_kernel_v5_async_vectorized",
        [&](dim3 grid, dim3 block, size_t smem_size) {
            flashattn_kernel_v5_async_vectorized<T, Br, Bc, D><<<grid, block, smem_size>>>(
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
                is_causal
            );
        },
        (Br + 2 * 2 * Bc) * D * sizeof(T)
    );

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_cu_q));
    CHECK_CUDA(cudaFree(d_cu_k));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O_ref);
    free(h_O_tmp);
    free(h_cu_q);
    free(h_cu_k);

    return 0;
}