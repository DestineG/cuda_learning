#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

void softmax_forward_cpu(float* input, float* output, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; ++i) {
        float *inp_row = input + i * num_classes;
        float *out_row = output + i * num_classes;

        // 最大值获取
        float max_val = -INFINITY;
        for (unsigned int j = 0; j < num_classes; ++j) {
            if (inp_row[j] > max_val) {
                max_val = inp_row[j];
            }
        }

        // 归一化参数计算
        float sum_exp = 0.0f;
        for (unsigned int j = 0; j < num_classes; ++j) {
            out_row[j] = expf(inp_row[j] - max_val);
            sum_exp += out_row[j];
        }

        // 归一化
        float norm = 1.0f / sum_exp;
        for (unsigned int j = 0; j < num_classes; ++j) {
            out_row[j] *= norm;
        }
    }
}

__global__ void softmax_forward_kernel1(float* input, float* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    float *inp_row = input + idx * num_classes;
    float *out_row = output + idx * num_classes;

    // 最大值获取
    float max_val = -INFINITY;
    for (unsigned int j = 0; j < num_classes; ++j) {
        if (inp_row[j] > max_val) {
            max_val = inp_row[j];
        }
    }

    // 归一化参数计算
    float sum_exp = 0.0f;
    for (unsigned int j = 0; j < num_classes; ++j) {
        out_row[j] = expf(inp_row[j] - max_val);
        sum_exp += out_row[j];
    }

    // 归一化
    float norm = 1.0f / sum_exp;
    for (unsigned int j = 0; j < num_classes; ++j) {
        out_row[j] *= norm;
    }
}

__global__ void softmax_forward_kernel2(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float shared[];
    unsigned int block_size = blockDim.x;
    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;

    // 每个线程计算一部分输入的最大值(步长为block_size)
    const float *inp_row = input + block_id * num_classes;
    float maxval = -INFINITY;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        maxval = fmaxf(maxval, inp_row[j]);
    }
    // 将每个线程的最大值存储到共享内存中
    shared[thread_id] = maxval;
    __syncthreads();
    // 归约求最大值
    for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared[thread_id] = fmaxf(shared[thread_id], shared[thread_id + stride]);
        }
        __syncthreads();
    }
    maxval = shared[0];

    // 每个线程计算一部分输入的指数和(步长为block_size)
    float *out_row = output + block_id * num_classes;
    float sum_exp = 0.0f;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        float val = expf(inp_row[j] - maxval);
        out_row[j] = val;
        sum_exp += val;
    }
    // 将每个线程的指数和存储到共享内存中
    shared[thread_id] = sum_exp;
    __syncthreads();
    // 归约求指数和
    for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared[thread_id] += shared[thread_id + stride];
        }
        __syncthreads();
    }
    sum_exp = shared[0];

    // 每个线程计算一部分输出的归一化值(步长为block_size)
    float norm = 1.0f / sum_exp;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        out_row[j] *= norm;
    }
}

__device__ float warpReduceMax(float val) {
    unsigned int warpSize = 32; // CUDA中warp的大小通常为32
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        // __shfl_down_sync函数用于在warp内进行数据交换，0xffffffff表示所有线程参与，val是当前线程的值，offset是交换的距离
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    unsigned int warpSize = 32; // CUDA中warp的大小通常为32
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        // __shfl_down_sync函数用于在warp内进行数据交换，0xffffffff表示所有线程参与，val是当前线程的值，offset是交换的距离
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel3(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float s_mem[]; 

    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int lane_id = thread_id % 32;

    const float *inp_row = input + block_id * num_classes;
    float *out_row = output + block_id * num_classes;

    // 求 max
    float maxval = -INFINITY;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        maxval = fmaxf(maxval, inp_row[j]);
    }

    // warp 内归约(此处函数使用了 __shfl_down_sync，它要求每个 warp 内的线程都参与计算)
    maxval = warpReduceMax(maxval);

    // 每个 warp 写一个值
    if (lane_id == 0) {
        s_mem[warp_id] = maxval;
    }
    __syncthreads();

    // 一个 warp 收尾(一个 warp 收尾意味着 block_size <= 1024)
    unsigned int num_warps = block_size / 32;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_mem[lane_id] : -INFINITY;
        val = warpReduceMax(val);

        if (lane_id == 0) {
            s_mem[0] = val;
        }
    }
    __syncthreads();
    maxval = s_mem[0];

    // 求 sum(exp)
    float sum_exp = 0.0f;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        float val = expf(inp_row[j] - maxval);
        out_row[j] = val;
        sum_exp += val;
    }

    // warp 内归约
    sum_exp = warpReduceSum(sum_exp);

    if (lane_id == 0) {
        s_mem[warp_id] = sum_exp;
    }
    __syncthreads();

    // 一个 warp 收尾
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_mem[lane_id] : 0.0f;
        val = warpReduceSum(val);

        if (lane_id == 0) {
            s_mem[0] = val;
        }
    }
    __syncthreads();

    sum_exp = s_mem[0];

    // normalize
    float norm = 1.0f / sum_exp;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        out_row[j] *= norm;
    }
}

__global__ void softmax_forward_kernel4(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float s_mem[]; 

    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int lane_id = thread_id % 32;

    const float *inp_row = input + block_id * num_classes;
    float *out_row = output + block_id * num_classes;

    // 求局部 max 和 sum(exp)
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        float val = inp_row[j];
        if (val > thread_max) {
            thread_sum = thread_sum * expf(thread_max - val) + 1.0f;
            thread_max = val;
        } else {
            thread_sum += expf(val - thread_max);
        }
    }

    // 求全局 max
    float block_max = warpReduceMax(thread_max);
    if (lane_id == 0) {
        s_mem[warp_id] = block_max;
    }
    __syncthreads();
    unsigned int num_warps = block_size / 32;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_mem[lane_id] : -INFINITY;
        val = warpReduceMax(val);
        if (lane_id == 0) {
            s_mem[0] = val;
        }
    }
    __syncthreads();
    block_max = s_mem[0];

    // 求全局 sum(exp)
    thread_sum = thread_sum * expf(thread_max - block_max);
    float block_sum = warpReduceSum(thread_sum);
    if (lane_id == 0) {
        s_mem[warp_id] = block_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_mem[lane_id] : 0.0f;
        val = warpReduceSum(val);
        if (lane_id == 0) {
            s_mem[0] = val;
        }
    }
    __syncthreads();
    block_sum = s_mem[0];

    // normalize
    float inv_sum = 1.0f / block_sum;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        out_row[j] = expf(inp_row[j] - block_max) * inv_sum;
    }
}

struct SumMax {
    float max_val;
    float sum_val;
};
__device__ SumMax merge_online(SumMax a, SumMax b) {
    if (a.max_val >= b.max_val) {
        return {a.max_val, a.sum_val + b.sum_val * expf(b.max_val - a.max_val)};
    } else {
        return {b.max_val, b.sum_val + a.sum_val * expf(a.max_val - b.max_val)};
    }
}
__device__ SumMax warpReduceOnline(SumMax sm) {
    for (int offset = 16; offset > 0; offset /= 2) {
        SumMax other;
        other.max_val = __shfl_down_sync(0xffffffff, sm.max_val, offset);
        other.sum_val = __shfl_down_sync(0xffffffff, sm.sum_val, offset);
        sm = merge_online(sm, other);
    }
    return sm;
}

__global__ void softmax_forward_kernel5(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float s_mem[]; 

    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int lane_id = thread_id % 32;

    const float *inp_row = input + block_id * num_classes;
    float *out_row = output + block_id * num_classes;

    // 求局部 max 和 sum(exp) (num_classes -> block_size)
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        float val = inp_row[j];
        if (val > thread_max) {
            thread_sum = thread_sum * expf(thread_max - val) + 1.0f;
            thread_max = val;
        } else {
            thread_sum += expf(val - thread_max);
        }
    }

    // 求全局 max 和 sum(exp) (block_size -> block_size/32)
    SumMax thread_sm{thread_max, thread_sum};
    SumMax block_sm = warpReduceOnline(thread_sm);
    if (lane_id == 0) {
        s_mem[warp_id * 2] = block_sm.max_val;
        s_mem[warp_id * 2 + 1] = block_sm.sum_val;
    }
    __syncthreads();
    // 一个 warp 收尾, block_size <= 1024 (block_size/32 -> 1)
    unsigned int num_warps = block_size / 32;
    if (warp_id == 0) {
        SumMax val{(lane_id < num_warps) ? s_mem[lane_id * 2] : -INFINITY, (lane_id < num_warps) ? s_mem[lane_id * 2 + 1] : 0.0f};
        val = warpReduceOnline(val);
        if (lane_id == 0) {
            s_mem[0] = val.max_val;
            s_mem[1] = val.sum_val;
        }
    }
    __syncthreads();
    float block_max = s_mem[0];
    float block_sum = s_mem[1];

    // normalize
    float inv_sum = 1.0f / block_sum;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        out_row[j] = expf(inp_row[j] - block_max) * inv_sum;
    }
}

// TODO: Vectorized Load/Store
__global__ void softmax_forward_kernel6(float* input, float* output, int batch_size, int num_classes);

void testSoftmaxCuDNN(float* d_input, float* d_output, int batch_size, int num_classes, const char* test_name) {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // 创建张量描述符
    // cuDNN 是为深度学习设计的，所以它习惯 4D 张量 (N, C, H, W)
    // 我们的数据是 (batch_size, num_classes)，对应 (N, C, 1, 1)
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_classes, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, desc, d_input, &beta, desc, d_output);
    cudaDeviceSynchronize();

    // 正式计时
    cudaEventRecord(start);
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, desc, d_input, &beta, desc, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) << test_name 
              << " | " << std::setw(15) << "cuDNN Library"
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms" 
              << std::endl;

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(handle);
}

void testSoftmax(void (*softmax_func)(float*, float*, int, int), float* input, float* output, int batch_size, int num_classes, const char* test_name) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行计算
    softmax_func(input, output, batch_size, num_classes);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算差值（毫秒）
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // 格式化输出
    std::cout << std::left << std::setw(30) << test_name 
              << " | Time: " << std::fixed << std::setprecision(4) << elapsed.count() << " ms" 
              << std::endl;
}

void testSoftmaxKernel(void (*kernel)(float*, float*, int, int), 
                       float* d_input, float* d_output, 
                       int batch_size, int num_classes, 
                       dim3 gridSize, dim3 blockSize, 
                       size_t sharedMemSize,
                       const char* test_name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热 (Warmup)
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, batch_size, num_classes);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, batch_size, num_classes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) << test_name 
              << " | Grid: " << std::setw(4) << gridSize.x 
              << " | Block: " << std::setw(4) << blockSize.x 
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms" 
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const unsigned int batch_size = 1024;
    const unsigned int num_classes = 8192;
    size_t size = batch_size * num_classes * sizeof(float);
    
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    for (unsigned int i = 0; i < batch_size * num_classes; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size);

    testSoftmax(softmax_forward_cpu, h_input, h_output, batch_size, num_classes, "CPU Softmax");

    {
        dim3 block(256);
        dim3 grid((batch_size + block.x - 1) / block.x);
        testSoftmaxKernel(softmax_forward_kernel1, d_input, d_output, batch_size, num_classes, 
                          grid, block, 0, "GPU Softmax Kernel 1");
    }

    {
        dim3 block(256); 
        dim3 grid(batch_size);
        size_t sharedMem = block.x * sizeof(float);
        testSoftmaxKernel(softmax_forward_kernel2, d_input, d_output, batch_size, num_classes, 
                          grid, block, sharedMem, "GPU Softmax Kernel 2");
    }

    {
        dim3 block(256); 
        dim3 grid(batch_size);
        size_t sharedMem = block.x * sizeof(float);
        testSoftmaxKernel(softmax_forward_kernel3, d_input, d_output, batch_size, num_classes, 
                          grid, block, sharedMem, "GPU Softmax Kernel 3");
    }

    {
        dim3 block(256); 
        dim3 grid(batch_size);
        size_t sharedMem = block.x * sizeof(float) / 32; // 每个 warp 写一个值
        testSoftmaxKernel(softmax_forward_kernel4, d_input, d_output, batch_size, num_classes, 
                          grid, block, sharedMem, "GPU Softmax Kernel 4");
    }

    {
        dim3 block(256); 
        dim3 grid(batch_size);
        size_t sharedMem = (block.x / 32) * 2 * sizeof(float); // 每个 warp 写一个 max 和 sum
        testSoftmaxKernel(softmax_forward_kernel5, d_input, d_output, batch_size, num_classes, 
                          grid, block, sharedMem, "GPU Softmax Kernel 5");
    }

    testSoftmaxCuDNN(d_input, d_output, batch_size, num_classes, "cuDNN Standard Accurate");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}