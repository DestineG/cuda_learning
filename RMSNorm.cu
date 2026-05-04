#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>

// 错误检查宏
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, reason:%s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// -----------------------------------------------------------------------------
// 算法实现
// -----------------------------------------------------------------------------
void RMSNorm_cpu(const float* input, const float* gamma, float* output, int batch, int hidden, float eps) {
    for (int i = 0; i < batch; ++i) {
        float rms = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            float val = input[i * hidden + j];
            rms += val * val;
        }
        rms = sqrtf(rms / hidden);
        for (int j = 0; j < hidden; ++j) {
            int idx = i * hidden + j;
            output[idx] = gamma[j] * input[idx] / (rms + eps);
        }
    }
}

__global__ void RMSNorm_kernel(const float* input, const float* gamma, float* output, int hidden, float eps) {
    __shared__ float shared_var[256]; 
    int row = blockIdx.x;
    float var = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = input[row * hidden + i];
        var += val * val;
    }
    shared_var[threadIdx.x] = var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared_var[threadIdx.x] += shared_var[threadIdx.x + stride];
        __syncthreads(); 
    }
    float rms = sqrtf(shared_var[0] / hidden);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        int idx = row * hidden + i;
        output[idx] = gamma[i] * input[idx] / (rms + eps);
    }
}

/*
next_kernel:
1. 计算 sum(var) 使用 warp shuffle
2. 读写 global memory, shared memory 使用 float4 向量化，减少指令数量
3. 目前 thread 的 compute throughput 偏低，考虑增加每个线程处理的元素数量，以提升计算密度
*/

// -----------------------------------------------------------------------------
// 抽象测试函数
// -----------------------------------------------------------------------------

// CPU 测试
void test_cpu(const float* input, const float* gamma, float* output, int batch, int hidden, float eps) {
    auto start = std::chrono::high_resolution_clock::now();
    
    RMSNorm_cpu(input, gamma, output, batch, hidden, eps);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << "[CPU Test]" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(4) << duration.count() << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// GPU 测试
void test_gpu(const float* d_input, const float* d_gamma, float* d_output, 
              const float* h_cpu_ref, int batch, int hidden, float eps, const char* kernel_name) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    RMSNorm_kernel<<<batch, 256>>>(d_input, d_gamma, d_output, hidden, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    
    RMSNorm_kernel<<<batch, 256>>>(d_input, d_gamma, d_output, hidden, eps);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float avg_msec = 0;
    CHECK_CUDA(cudaEventElapsedTime(&avg_msec, start, stop));

    float* h_gpu_res = (float*)malloc(batch * hidden * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_gpu_res, d_output, batch * hidden * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_err = 0;
    for (int i = 0; i < batch * hidden; ++i) {
        max_err = fmax(max_err, fabsf(h_cpu_ref[i] - h_gpu_res[i]));
    }

    // 计算指标
    double total_bytes = (double)batch * hidden * 4 * sizeof(float);
    double bandwidth = (total_bytes / (avg_msec * 1e-3)) / 1e9;

    std::cout << kernel_name << std::endl;
    std::cout << "  Time:       " << std::fixed << std::setprecision(4) << avg_msec << " ms" << std::endl;
    std::cout << "  Bandwidth:  " << bandwidth << " GB/s" << std::endl;
    std::cout << "  Max Error:  " << max_err << std::endl;
    std::cout << "  Status:     " << (max_err < 1e-4 ? "PASS" : "FAIL") << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    free(h_gpu_res);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// -----------------------------------------------------------------------------
// 主程序
// -----------------------------------------------------------------------------
int main() {
    const int batch = 1024;
    const int hidden = 4096;
    const float eps = 1e-5f;

    size_t matrix_bytes = (size_t)batch * hidden * sizeof(float);
    size_t gamma_bytes = (size_t)hidden * sizeof(float);

    float* h_in      = (float*)malloc(matrix_bytes);
    float* h_gamma   = (float*)malloc(gamma_bytes);
    float* h_out_cpu = (float*)malloc(matrix_bytes);

    if (h_in == nullptr || h_gamma == nullptr || h_out_cpu == nullptr) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return -1;
    }

    for (int i = 0; i < batch * hidden; ++i) {
        h_in[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < hidden; ++i) {
        h_gamma[i] = (float)rand() / RAND_MAX;
    }

    float *d_in, *d_gamma, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, gamma_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, matrix_bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, gamma_bytes, cudaMemcpyHostToDevice));

    std::cout << "Starting Tests (Batch=" << batch << ", Hidden=" << hidden << ")\n" << std::endl;
    test_cpu(h_in, h_gamma, h_out_cpu, batch, hidden, eps);
    test_gpu(d_in, d_gamma, d_out, h_out_cpu, batch, hidden, eps, "RMSNormShared_kernel");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_out));
    
    free(h_in);
    free(h_gamma);
    free(h_out_cpu);

    return 0;
}