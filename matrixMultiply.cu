#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cuda_pipeline.h>

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
    const int TILE_SIZE = 32;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x; 
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; 
    float value = 0;

    unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (unsigned int t = 0; t < numTiles; ++t)
    {
        int colOffset = t * TILE_SIZE + threadIdx.x;
        int rowOffset = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && colOffset < K) ? A[row * K + colOffset] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (rowOffset < K && col < P) ? B[rowOffset * P + col] : 0.0f;
        
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < P) C[row * P + col] = value;
}

__device__ __forceinline__ void loadTileAAsync(float* sharedA, const float* globalA, int row, int colOffset, int M, int K, int TILE_SIZE) {
    if (row < M && colOffset < K) {
        __pipeline_memcpy_async(sharedA, &globalA[row * K + colOffset], sizeof(float));
    } else {
        *sharedA = 0.0f;
    }
}

__device__ __forceinline__ void loadTileBAsync(float* sharedB, const float* globalB, int rowOffset, int col, int K, int P, int TILE_SIZE) {
    if (rowOffset < K && col < P) {
        __pipeline_memcpy_async(sharedB, &globalB[rowOffset * P + col], sizeof(float));
    } else {
        *sharedB = 0.0f;
    }
}

__global__ void asyncTileMatrixMultiply(float *A, float *B, float *C, int M, int K, int P) {
    const int TILE_SIZE = 32;
    __shared__ float tileA[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[2][TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    float value = 0;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    loadTileAAsync(&tileA[0][threadIdx.y][threadIdx.x], A, row, 0 * TILE_SIZE + threadIdx.x, M, K, TILE_SIZE);
    loadTileBAsync(&tileB[0][threadIdx.y][threadIdx.x], B, 0 * TILE_SIZE + threadIdx.y, col, K, P, TILE_SIZE);
    __pipeline_commit();

    for (int t = 1; t < numTiles; ++t) {
        int curr = (t - 1) % 2;
        int next = t % 2;

        loadTileAAsync(&tileA[next][threadIdx.y][threadIdx.x], A, row, t * TILE_SIZE + threadIdx.x, M, K, TILE_SIZE);
        loadTileBAsync(&tileB[next][threadIdx.y][threadIdx.x], B, t * TILE_SIZE + threadIdx.y, col, K, P, TILE_SIZE);
        __pipeline_commit();

        __pipeline_wait_prior(1); 
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[curr][threadIdx.y][k] * tileB[curr][k][threadIdx.x];
        }
        __syncthreads();
    }

    __pipeline_wait_prior(0); 
    __syncthreads();
    
    int last = (numTiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        value += tileA[last][threadIdx.y][k] * tileB[last][k][threadIdx.x];
    }

    if (row < M && col < P) C[row * P + col] = value;
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

    // 预热
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    size_t size_C = M * P * sizeof(float);
    float *h_C = (float*)malloc(size_C);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << std::left << std::setw(20) << name 
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | C[0]: " << h_C[0] << std::endl;

    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main()
{
    const int M = 2048;
    const int K = 2048;
    const int P = 2048;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * P * sizeof(float);
    size_t size_C = M * P * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * P; ++i) h_B[i] = 0.01f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    std::cout << "Matrix Shape: A(" << M << "x" << K << ") * B(" << K << "x" << P << ")" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    testMatrixKernel(matrixMultiply, d_A, d_B, d_C, M, K, P, "Naive GEMM");
    testMatrixKernel(tileMatrixMultiply, d_A, d_B, d_C, M, K, P, "Tiled GEMM");
    testMatrixKernel(asyncTileMatrixMultiply, d_A, d_B, d_C, M, K, P, "Async Tiled GEMM");

    std::cout << "------------------------------------------------------------" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);

    return 0;
}