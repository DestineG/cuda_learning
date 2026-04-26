#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int K, int P)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < P)
    {
        float value = 0;
        for (int k = 0; k < K; ++k)
        {
            value += A[row * K + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

__global__ void tileMatrixMultiply(float *A, float *B, float *C, int M, int K, int P)
{
    const int TILE_SIZE = 32;   // tile 的大小必须等于 blockDim.x 和 blockDim.y 每个 block 负责计算矩阵 C 的一个 TILE_SIZE x TILE_SIZE 的 tile
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; 
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; 
    float value = 0;

    unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int colOffset, rowOffset;
    for (unsigned int t = 0; t < numTiles; ++t)
    {
        // 加载矩阵 A 的 tile
        colOffset = t * TILE_SIZE + threadIdx.x;
        if (row < M && colOffset < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + colOffset];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        // 加载矩阵 B 的 tile
        rowOffset = t * TILE_SIZE + threadIdx.y;
        if (rowOffset < K && col < P)
            tileB[threadIdx.y][threadIdx.x] = B[rowOffset * P + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        // 计算 tileA @ tileB
        for (unsigned int k = 0; k < TILE_SIZE; ++k)
        {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 同步以确保所有线程完成计算后再加载下一个 tile，否则跑得快的线程可能会覆盖共享内存中的 tileA 和 tileB 导致其他线程读取到错误的数据
         __syncthreads();
    }

    if (row < M && col < P)
        C[row * P + col] = value;
}

float testMatrixKernel(void (*kernel)(float*, float*, float*, int, int, int), 
                      float* d_A, float* d_B, float* d_C, 
                      int M, int K, int P, const char* name) 
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(32, 32); 
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    size_t size_C = M * P * sizeof(float);
    float *h_C = (float*)malloc(size_C);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << std::left << std::setw(15) << name 
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | C[0]: " << h_C[0] << std::endl;

    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main()
{
    // 定义非对称矩阵维度
    const int M = 1024;
    const int K = 1024;
    const int P = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * P * sizeof(float);
    size_t size_C = M * P * sizeof(float);

    // 分配主机内存
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * P; ++i) h_B[i] = 2.0f;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 拷贝输入数据
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    std::cout << "Matrix Shape: A(" << M << "x" << K << ") * B(" << K << "x" << P << ")" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // 运行测试
    testMatrixKernel(matrixMultiply, d_A, d_B, d_C, M, K, P, "Naive GEMM");
    testMatrixKernel(tileMatrixMultiply, d_A, d_B, d_C, M, K, P, "Tiled GEMM");

    std::cout << "------------------------------------------------------------" << std::endl;

    // 释放资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);

    return 0;
}