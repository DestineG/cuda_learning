// Grid 的维度设计为 ((seq_len + Br - 1) / Br, num_heads, batch)
template<typename T, int Br, int Bc, int D>
__global__ void flashattn_kernel(
    const T* __restrict__ Q,           // (Total_Q, H, D)
    const T* __restrict__ K,           // (Total_KV, H_k, D)
    const T* __restrict__ V,           // (Total_KV, H_k, D)
    T* __restrict__ O,                 // (Total_Q, H, D)
    const int* __restrict__ cu_seqlens_q, // (B + 1)
    const int* __restrict__ cu_seqlens_k, // (B + 1)
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float scale,
    const int num_heads,
    const int num_heads_k,
    // 步长参数用于处理各种 Tensor 内存排布
    const int q_stride_token, const int q_stride_head,
    const int k_stride_token, const int k_stride_head,
    const int v_stride_token, const int v_stride_head,
    const int o_stride_token, const int o_stride_head,
    bool is_causal
) {
    int seq_idx = blockIdx.z;
    int head_idx = blockIdx.y;

    const int q_start = cu_seqlens_q[seq_idx];
    const int kv_start = cu_seqlens_k[seq_idx];
    const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
    const int kv_len = cu_seqlens_k[seq_idx + 1] - kv_start;

    // 计算当前 block 负责处理的 Q 的起始位置
    int q_block_start = blockIdx.x * Br;
    if (q_block_start >= q_len) return; // 超出 Q 长度范围，退出
    int q_block_end = min(q_block_start + Br, q_len);
    const T* q_ptr = (
                    Q                                       // Q 的起始地址
                    + q_start * q_stride_token              // 跳过前面序列的 Q
                    + head_idx * q_stride_head              // 定位到当前 head
                    + q_block_start * q_stride_token        // 定位到当前 block 内的起始 Q
    );

    // 计算当前 block 负责处理的 K 和 V 的起始位置
    int kv_head_idx = head_idx / (num_heads / num_heads_k); // GQA 中 K/V 的 head 索引
    int kv_block_start = 0; // K/V 从头开始处理
    const T* k_ptr = (
                    K                                       // K 的起始地址
                    + kv_start * k_stride_token             // 跳过前面序列的 K
                    + kv_head_idx * k_stride_head           // 定位到当前 head
                    + kv_block_start * k_stride_token       // 定位到当前 block 内的起始 K
    );
    const T* v_ptr = (
                    V                                       // V 的起始地址
                    + kv_start * v_stride_token             // 跳过前面序列的 V
                    + kv_head_idx * v_stride_head           // 定位到当前 head
                    + kv_block_start * v_stride_token       // 定位到当前 block 内的起始 V
    );

    // 计算输出 O 的起始位置
    T* o_ptr = (
                    O                                       // O 的起始地址
                    + q_start * o_stride_token              // 跳过前面序列的 O
                    + head_idx * o_stride_head              // 定位到当前 head
                    + q_block_start * o_stride_token        // 定位到当前 block 内的起始 O
    );

    // shared memory 分配 <<<Grid, Block, sizeof(T) * (Br * D + 2 * Bc * D)>>>，布局为 [s_q(Br, D), s_k(Bc, D), s_v(Bc, D)]
    extern __shared__ char smem[];
    T* s_q = reinterpret_cast<T*>(smem); // (Br, D)
    T* s_k = s_q + Br * D;                // (Bc, D)
    T* s_v = s_k + Bc * D;                // (Bc, D)

}