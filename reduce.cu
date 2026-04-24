#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void initInput(float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) input[idx] = 1.0f;
}

__global__ void reduceGmem(float *input, float *output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 强制写全局内存
    volatile float *startPtr = input + blockIdx.x * blockDim.x;

    if (blockDim.x > 512 && tid < 512 && (idx + 512) < n) startPtr[tid] += startPtr[tid + 512];
    __syncthreads();
    if (blockDim.x > 256 && tid < 256 && (idx + 256) < n) startPtr[tid] += startPtr[tid + 256];
    __syncthreads();
    if (blockDim.x > 128 && tid < 128 && (idx + 128) < n) startPtr[tid] += startPtr[tid + 128];
    __syncthreads();
    if (blockDim.x > 64 && tid < 64 && (idx + 64) < n) startPtr[tid] += startPtr[tid + 64];
    __syncthreads();

    if (tid < 32) {
        if (blockDim.x > 32 && (idx + 32) < n) startPtr[tid] += startPtr[tid + 32];
        if (blockDim.x > 16 && (idx + 16) < n) startPtr[tid] += startPtr[tid + 16];
        if (blockDim.x > 8 && (idx + 8) < n) startPtr[tid] += startPtr[tid + 8];
        if (blockDim.x > 4 && (idx + 4) < n) startPtr[tid] += startPtr[tid + 4];
        if (blockDim.x > 2 && (idx + 2) < n) startPtr[tid] += startPtr[tid + 2];
        if (blockDim.x > 1 && (idx + 1) < n) startPtr[tid] += startPtr[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = startPtr[0];
}

__global__ void reduceSmem(float *input, float *output, int n) {
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    if (idx >= n) return;

    if (blockDim.x > 512 && tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (blockDim.x > 256 && tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (blockDim.x > 128 && tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (blockDim.x > 64 && tid < 64)   sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        // 使用 volatile 确保每次访问都从共享内存中读取最新值(否则可能从寄存器中读取过时值)
        volatile float *vsmem = sdata;
        if (blockDim.x > 32) vsmem[tid] += vsmem[tid + 32];
        if (blockDim.x > 16) vsmem[tid] += vsmem[tid + 16];
        if (blockDim.x > 8)  vsmem[tid] += vsmem[tid + 8];
        if (blockDim.x > 4)  vsmem[tid] += vsmem[tid + 4];
        if (blockDim.x > 2)  vsmem[tid] += vsmem[tid + 2];
        if (blockDim.x > 1)  vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

float testKernel(void (*kernel)(float*, float*, int), float* d_input, float* d_output, int n, int numBlocks, int blockSize, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 重新初始化数据，避免上一次测试的残留结果
    initInput<<<numBlocks, blockSize>>>(d_input, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float *h_output = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < numBlocks; i++) sum += h_output[i];

    std::cout << std::left << std::setw(12) << name 
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | Sum: " << sum << std::endl;

    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int n = 1 << 26;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));

    std::cout << "Data Size: " << n << " (" << (n * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    testKernel(reduceGmem, d_input, d_output, n, numBlocks, blockSize, "Global Mem");
    testKernel(reduceSmem, d_input, d_output, n, numBlocks, blockSize, "Shared Mem");

    std::cout << "--------------------------------------------------------" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}