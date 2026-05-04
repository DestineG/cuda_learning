#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cublas_v2.h>
#include <assert.h>

void print_performance(const char* name, float milliseconds, double tflops) {
    std::cout << std::left  << std::setw(45) << name 
              << std::left  << std::setw(10) << " | Time: " 
              << std::right << std::setw(12)  << std::fixed << std::setprecision(4) << milliseconds 
              << std::left  << std::setw(5)  << " ms" 
              << std::left  << std::setw(12) << " | TFLOPS: " 
              << std::right << std::setw(8)  << std::fixed << std::setprecision(4) << tflops 
              << std::endl;
}

void matrixMultiply_cpu(float *A, float *B, float *C, int M, int K, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float valA = A[i * K + k];
                float valB = B[j * K + k];
                sum += valA * valB;
            }
            C[i * P + j] = sum;
        }
    }
}

void testCPUKernel(float *A, float *B, float *C, int M, int K, int P, const char *name) {
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply_cpu(A, B, C, M, K, P);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    double ops = 2.0 * M * K * P;
    double tflops = (ops / (duration.count() / 1000.0)) / 1e12;

    print_performance(name, duration.count(), tflops);
}

void testCublasGEMM(float *d_A, float *d_B, float *d_C, int M, int K, int P) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, M, K, &alpha, d_B, P, d_A, K, &beta, d_C, P);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    // cuBLAS 是列优先，所以计算 C = A * B (行优先) 相当于计算 C^T = B^T * A^T (列优先)
    // 这里传入的参数顺序经过了调整以匹配行优先结果
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, M, K, &alpha, d_B, P, d_A, K, &beta, d_C, P);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double ops = 2.0 * M * K * P;
    double tflops = (ops / (milliseconds / 1000.0)) / 1e12;

    print_performance("cuBLAS Standard", milliseconds, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
}

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int K, int P) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < P)
    {
        float value = 0;
        for (int k = 0; k < K; ++k)
        {
            // Warp 访存广播: 同一个 Warp 内的线程访问相同的 A[row * K + k]，可以通过广播机制减少访存带宽压力
            // Warp 访存合并: 同一个 Warp 内的线程访问连续的 B[k * P + col]，可以通过访存合并机制提高访存效率
            value += A[row * K + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

__global__ void matmulFloat4_kernel(float *A, float *B_col, float *C, int M, int K, int P){
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < P)
    {
        float value = 0;
        for (int k = 0; k < K; k += 4)
        {
            // 将 A 和 B_col 的访问转换为 float4 类型，以利用向量化加载和计算(K 必须是 4 的倍数)
            float4 a_vec = reinterpret_cast<float4*>(A + row * K + k)[0];
            float4 b_vec = reinterpret_cast<float4*>(B_col + col * K + k)[0];

            // 计算 a_vec 和 b_vec 的点积
            value += a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
        }
        C[row * P + col] = value;
    }
}

__global__ void matmulshared_kernel(float *A, float *B, float *C, int M, int K, int P) {
    const int TILE_SIZE = 32;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // (blockDim.x, blockDim.y) = (TILE_SIZE, TILE_SIZE)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float value = 0;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        // 将 A 的子矩阵加载到共享内存中
        if (row < M && tiledCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // 将 B 的子矩阵加载到共享内存中
        if (col < P && tiledRow < K)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * P + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算当前 tile 的乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            // Shared Memory 由 32 个 bank 组成，每个 bank 的宽度为 4 字节(float)，bank 之间低位地址连续
            // Bank Conflict: 同一个 warp 内的线程访问了共享内存中相同的 bank 导致 warp 访存串行化，降低性能
            // 此处的 tileA[threadIdx.y][k] 意味着同一个 warp 内的线程访问了 tileA 中相同位置的元素，正常来说会导致 bank conflict，但 warp 访存广播机制可以将访问相同地址的请求合并为一次访问，从而避免 bank conflict 的性能损失
            // tileB[k][threadIdx.x] 意味着同一个 warp 内的线程访问了 tileB 中同一行但是不同列的元素，也就是每个线程刚好访问不同 bank 的元素，因此不会产生 bank conflict，并且由于 warp 中线程访问的地址是连续的，所以也可能触发 warp 访存合并机制，进一步提高访存效率
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * P + col] = value;
}

__global__ void matmulsharedFloat4_kernel(float *A, float *B, float *C, int M, int K, int P) {
const int TILE_SIZE = 32;
    __shared__ float4 tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float4 tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float value = 0;

    for (int t = 0; t < (K + (TILE_SIZE * 4) - 1) / (TILE_SIZE * 4); ++t) {        
        // A (行优先)
        int tiledColA = t * (TILE_SIZE * 4) + tx * 4; 
        if (row < M && tiledColA < K) {
            tileA[ty][tx] = reinterpret_cast<float4*>(A + row * K + tiledColA)[0];
        } else {
            tileA[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // B (列优先)
        int tiledRowB = t * (TILE_SIZE * 4) + ty * 4;
        if (col < P && tiledRowB < K) {
            tileB[ty][tx] = reinterpret_cast<float4*>(B + col * K + tiledRowB)[0];
        } else {
            tileB[ty][tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float4 a = tileA[ty][k];
            float4 b = tileB[k][tx];
            value += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = value;
    }
}

void testMatrixKernel(
    void (*kernel)(float *, float *, float *, int, int, int),
    float *d_A, float *d_B, float *d_C, 
    int M, int K, int P,
    int blockX, int blockY,
    const char *name) 
{
    dim3 blockSize(blockX, blockY);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    
    // 检查启动配置错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!! [" << name << "] Launch Error (Warm-up): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();
    
    // 检查运行阶段错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!! [" << name << "] Execution Error (Warm-up): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!! [" << name << "] Launch Error (Timed): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double ops = 2.0 * M * K * P;
    double tflops = (ops / (milliseconds / 1000.0)) / 1e12;

    print_performance(name, milliseconds, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 计算 每个线程的搬运次数(每个线程每次搬运 1 float)
template <int ThreadDimx, int ThreadDimy,int ThreadTileX, int ThreadTileY, int TILE_K>
__global__ void matmulsharedThreadTiling_kernel(float *A, float *B, float *C, int M, int K, int P) {
    // assert 保证整除 (ThreadDimy * ThreadTileY * TILE_K) % (ThreadDimy * ThreadDimx) == 0, 以去掉搬运次数中的余数，简化填充 TileA 逻辑
    static_assert(
        (ThreadDimy * ThreadTileY * TILE_K) % (ThreadDimy * ThreadDimx) == 0,
        "ThreadTileY * TILE_K must be divisible by ThreadDimx");
    // assert 保证整除 (ThreadTileX * ThreadDimx * TILE_K) % (ThreadDimy * ThreadDimx) == 0, 以去掉搬运次数中的余数，简化填充 TileB 逻辑
    static_assert(
        (ThreadTileX * ThreadDimx * TILE_K) % (ThreadDimy * ThreadDimx) == 0,
        "ThreadTileX * TILE_K must be divisible by ThreadDimy");

    __shared__ float tileA[ThreadTileY * ThreadDimy][TILE_K + 1];
    __shared__ float tileB[TILE_K][ThreadTileX * ThreadDimx];

    float accum[ThreadTileY][ThreadTileX];
    #pragma unroll
    for (int i = 0; i < ThreadTileY; ++i) {
        for (int j = 0; j < ThreadTileX; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    for (int t = 0; t < (K + TILE_K -1) / TILE_K; ++t) {
        // 填充 tileA
        int globalARow = blockIdx.y * (ThreadDimy * ThreadTileY);
        int globalACol = t * TILE_K;
        int numLoadsA = (ThreadDimy * ThreadTileY * TILE_K) / (ThreadDimy * ThreadDimx);
        #pragma unroll
        for (int i = 0; i < numLoadsA; ++i) {
            int idx = threadIdx.y * ThreadDimx + threadIdx.x + i * (ThreadDimy * ThreadDimx);
            int localRow = idx / TILE_K;
            int localCol = idx % TILE_K;
            int globalRow = globalARow + localRow;
            int globalCol = globalACol + localCol;
            if (globalRow < M && globalCol < K) {
                tileA[localRow][localCol] = A[globalRow * K + globalCol];
            } else {
                tileA[localRow][localCol] = 0.0f;
            }
        }

        // 填充 tileB
        int globalBRow = t * TILE_K;
        int globalBCol = blockIdx.x * (ThreadDimx * ThreadTileX);
        int numLoadsB = (ThreadTileX * ThreadDimx * TILE_K) / (ThreadDimy * ThreadDimx);
        #pragma unroll
        for (int i = 0; i < numLoadsB; ++i) {
            int idx = threadIdx.y * ThreadDimx + threadIdx.x + i * (ThreadDimy * ThreadDimx);
            int localRow = idx / (ThreadTileX * ThreadDimx);
            int localCol = idx % (ThreadTileX * ThreadDimx);
            int globalRow = globalBRow + localRow;
            int globalCol = globalBCol + localCol;
            if (globalRow < K && globalCol < P) {
                tileB[localRow][localCol] = B[globalRow * P + globalCol];
            } else {
                tileB[localRow][localCol] = 0.0f;
            }
        }

        __syncthreads();

        // 外积计算
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float regA[ThreadTileY];
            float regB[ThreadTileX];
            #pragma unroll
            for (int i = 0; i < ThreadTileY; ++i) regA[i] = tileA[threadIdx.y * ThreadTileY + i][k];
            #pragma unroll
            for (int j = 0; j < ThreadTileX; ++j) regB[j] = tileB[k][threadIdx.x * ThreadTileX + j];
            #pragma unroll
            for (int i = 0; i < ThreadTileY; ++i) {
                for (int j = 0; j < ThreadTileX; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回结果
    int globalCTileRow = blockIdx.y * (ThreadDimy * ThreadTileY) + (threadIdx.y * ThreadTileY);
    int globalCTileCol = blockIdx.x * (ThreadDimx * ThreadTileX) + (threadIdx.x * ThreadTileX);
    for (int i = 0; i < ThreadTileY; ++i) {
        for (int j = 0; j < ThreadTileX; ++j) {
            int row = globalCTileRow + i;
            int col = globalCTileCol + j;
            if (row < M && col < P) {
                C[row * P + col] = accum[i][j];
            }
        }
    }
}

template <int ThreadDimx, int ThreadDimy, int ThreadTileX, int ThreadTileY, int TILE_K>
__global__ void matmulsharedThreadTilingFloat4_kernel(float *A, float *B, float *C, int M, int K, int P) {
    // 确保 Tile 宽度是 4 的倍数，且搬运总量能被 (线程数 * 4) 整除
    static_assert(TILE_K % 4 == 0, "TILE_K must be a multiple of 4 for float4 loading");
    static_assert((ThreadTileX * ThreadDimx) % 4 == 0, "TileB width must be a multiple of 4");
    
    // 确保搬运次数是整数，避免处理搬运剩余元素的复杂逻辑
    static_assert(
        (ThreadDimy * ThreadTileY * TILE_K) % (ThreadDimy * ThreadDimx * 4) == 0,
        "Total elements in TileA must be divisible by (num_threads * 4)");
    static_assert(
        (ThreadTileX * ThreadDimx * TILE_K) % (ThreadDimy * ThreadDimx * 4) == 0,
        "Total elements in TileB must be divisible by (num_threads * 4)");

    __shared__ float tileA[ThreadTileY * ThreadDimy][TILE_K + 1];
    __shared__ float tileB[TILE_K][ThreadTileX * ThreadDimx];

    float accum[ThreadTileY][ThreadTileX];
    #pragma unroll
    for (int i = 0; i < ThreadTileY; ++i) {
        for (int j = 0; j < ThreadTileX; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    int threadsPerBlock = ThreadDimy * ThreadDimx;
    int tid = threadIdx.y * ThreadDimx + threadIdx.x;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        
        // 填充 tileA
        int globalARowBase = blockIdx.y * (ThreadDimy * ThreadTileY);
        int globalAColBase = t * TILE_K;
        int numLoadsA = (ThreadDimy * ThreadTileY * TILE_K) / (threadsPerBlock * 4);

        #pragma unroll
        for (int i = 0; i < numLoadsA; ++i) {
            int idx = tid + i * threadsPerBlock;
            int localRow = (idx * 4) / TILE_K;
            int localCol = (idx * 4) % TILE_K;
            int globalRow = globalARowBase + localRow;
            int globalCol = globalAColBase + localCol;

            if (globalRow < M && globalCol < K) {
                // 要求 A 的地址偏移必须对齐 128-bit
                float4 tmp = reinterpret_cast<float4*>(&A[globalRow * K + globalCol])[0];
                tileA[localRow][localCol]     = tmp.x;
                tileA[localRow][localCol + 1] = tmp.y;
                tileA[localRow][localCol + 2] = tmp.z;
                tileA[localRow][localCol + 3] = tmp.w;
            } else {
                tileA[localRow][localCol] = tileA[localRow][localCol+1] = 
                tileA[localRow][localCol+2] = tileA[localRow][localCol+3] = 0.0f;
            }
        }

        // 填充 tileB
        int globalBRowBase = t * TILE_K;
        int globalBColBase = blockIdx.x * (ThreadDimx * ThreadTileX);
        int numLoadsB = (TILE_K * ThreadDimx * ThreadTileX) / (threadsPerBlock * 4);

        #pragma unroll
        for (int i = 0; i < numLoadsB; ++i) {
            int idx = tid + i * threadsPerBlock;
            int localRow = (idx * 4) / (ThreadDimx * ThreadTileX);
            int localCol = (idx * 4) % (ThreadDimx * ThreadTileX);
            int globalRow = globalBRowBase + localRow;
            int globalCol = globalBColBase + localCol;

            if (globalRow < K && globalCol < P) {
                float4 tmp = reinterpret_cast<float4*>(&B[globalRow * P + globalCol])[0];
                tileB[localRow][localCol]     = tmp.x;
                tileB[localRow][localCol + 1] = tmp.y;
                tileB[localRow][localCol + 2] = tmp.z;
                tileB[localRow][localCol + 3] = tmp.w;
            } else {
                tileB[localRow][localCol] = tileB[localRow][localCol+1] = 
                tileB[localRow][localCol+2] = tileB[localRow][localCol+3] = 0.0f;
            }
        }

        __syncthreads();

        // 计算
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float regA[ThreadTileY];
            float regB[ThreadTileX];
            #pragma unroll
            for (int i = 0; i < ThreadTileY; ++i) regA[i] = tileA[threadIdx.y * ThreadTileY + i][k];
            #pragma unroll
            for (int j = 0; j < ThreadTileX; ++j) regB[j] = tileB[k][threadIdx.x * ThreadTileX + j];
            
            #pragma unroll
            for (int i = 0; i < ThreadTileY; ++i) {
                for (int j = 0; j < ThreadTileX; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回
    int globalCTileRow = blockIdx.y * (ThreadDimy * ThreadTileY) + (threadIdx.y * ThreadTileY);
    int globalCTileCol = blockIdx.x * (ThreadDimx * ThreadTileX) + (threadIdx.x * ThreadTileX);
    for (int i = 0; i < ThreadTileY; ++i) {
        for (int j = 0; j < ThreadTileX; ++j) {
            int row = globalCTileRow + i;
            int col = globalCTileCol + j;
            if (row < M && col < P) {
                C[row * P + col] = accum[i][j];
            }
        }
    }
}

/*
summary:
1. 只要 warp 内部线程之间操作是密集的，那么就不容易触发 BC
2. 研究为何 [TILE_K + 1] 可以避免 bank conflict
*/

/*
next kernel idea:
1. 动态搜索最佳 Tile 大小和线程块配置以最大化性能，适应不同矩阵大小和 GPU 架构
2. global memory -> shared memory -> registers 的向量化加载，减少访存指令数量
3. 双缓冲技术：在计算当前 tile 的同时预加载下一个 tile，隐藏内存访问延迟
4. 找找 Tensor Core 的使用机会，进一步提升性能
5. 在计算外积的时候顺序为 读取(shared -> registers) -> 计算 串行执行，可以在计算的时候同时从 shared memory 加载下一组数据到寄存器中
6. 考虑每个 warp 协同重新设计 tile 的布局和访问模式，以进一步减少 bank conflict 和提高数据重用率
*/

void testThreadTilingKernel(
    void (*kernel)(float *, float *, float *, int, int, int),
    float *d_A, float *d_B, float *d_C, 
    int M, int K, int P,
    int ThreadDimx, int ThreadDimy,
    int ThreadTileX, int ThreadTileY,
    const char *name) 
{
    // 计算 Block Tile
    int blockTileX = ThreadDimx * ThreadTileX;
    int blockTileY = ThreadDimy * ThreadTileY;

    dim3 blockSize(ThreadDimx, ThreadDimy);
    
    // 根据 Block Tile 计算 Grid
    dim3 gridSize((P + blockTileX - 1) / blockTileX, 
                  (M + blockTileY - 1) / blockTileY);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    
    // 检查配置
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!! [" << name << "] Launch Error (Warm-up): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();
    
    // 计时运行
    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, P);
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!! [" << name << "] Launch Error (Timed): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算 TFLOPS
    double ops = 2.0 * M * K * P;
    double tflops = (ops / (milliseconds / 1000.0)) / 1e12;

    print_performance(name, milliseconds, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

bool verify_result(float *host_C, float *gpu_res_C, int M, int P, float epsilon = 1e-3) {
    for (int i = 0; i < M * P; ++i) {
        float diff = std::abs(host_C[i] - gpu_res_C[i]);
        float rel_err = diff / (std::abs(host_C[i]) + 1e-9f); // 防止除零
        if (rel_err > epsilon && diff > 1e-4) { // 同时考虑绝对误差和相对误差
            std::cout << "Result mismatch at index " << i 
                      << " CPU: " << host_C[i] << " GPU: " << gpu_res_C[i] 
                      << " RelErr: " << rel_err << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 2048;
    const int K = 2048;
    const int P = 2048;
    
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * P * sizeof(float);
    const size_t size_C = M * P * sizeof(float);

    float *h_A      = (float *)malloc(size_A);
    float *h_B_row  = (float *)malloc(size_B);
    float *h_B_col  = (float *)malloc(size_B);
    float *h_C_ref  = (float *)malloc(size_C);
    float *h_C_gpu  = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < P; ++c) {
            float val = rand() / (float)RAND_MAX;
            h_B_row[r * P + c] = val;
            h_B_col[c * K + r] = val;
        }
    }

    float *d_A, *d_B_row, *d_B_col, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B_row, size_B);
    cudaMalloc(&d_B_col, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_row, h_B_row, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col, h_B_col, size_B, cudaMemcpyHostToDevice);

    std::cout << "Matrix Size: " << M << "x" << K << "x" << P << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // 计算 CPU 基准结果
    testCPUKernel(h_A, h_B_col, h_C_ref, M, K, P, "CPU Naive GEMM");
    std::cout << std::string(85, '-') << std::endl;

    // 验证逻辑辅助 Lambda
    auto verify = [&](const char* kernelName) {
        cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
        
        std::cout << std::left << std::setw(45) << "   >> [Verification]" 
                << std::left << std::setw(10) << " | Result: ";
        
        if (verify_result(h_C_ref, h_C_gpu, M, P)) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED!" << std::endl;
        }
        
        std::cout << std::string(85, '-') << std::endl;
    };
    
    // GPU Naive
    testMatrixKernel(matmul_kernel, d_A, d_B_row, d_C, M, K, P, 32, 32, "GPU Naive GEMM");
    verify("GPU Naive GEMM");

    // GPU Float4
    testMatrixKernel(matmulFloat4_kernel, d_A, d_B_col, d_C, M, K, P, 32, 32, "GPU Float4 GEMM");
    verify("GPU Float4 GEMM");

    // GPU Shared Memory
    testMatrixKernel(matmulshared_kernel, d_A, d_B_row, d_C, M, K, P, 32, 32, "GPU Shared Memory GEMM");
    verify("GPU Shared Memory GEMM");

    // GPU Shared Memory Float4
    testMatrixKernel(matmulsharedFloat4_kernel, d_A, d_B_col, d_C, M, K, P, 32, 32, "GPU Shared Memory Float4 GEMM");
    verify("GPU Shared Memory Float4 GEMM");

    // GPU Thread Tiling
    {
        const int ThreadDimx = 16;
        const int ThreadDimy = 16;
        const int ThreadTileX = 8;
        const int ThreadTileY = 8;
        const int TILE_K = 16;
        testThreadTilingKernel(
            matmulsharedThreadTiling_kernel<ThreadDimx, ThreadDimy, ThreadTileX, ThreadTileY, TILE_K>, 
            d_A, d_B_row, d_C, M, K, P, 
            ThreadDimx, ThreadDimy, ThreadTileX, ThreadTileY, 
            "GPU Shared Memory Thread Tiling GEMM"
        );
        verify("GPU Shared Memory Thread Tiling GEMM");
    }

    {
        const int ThreadDimx = 16;
        const int ThreadDimy = 16;
        const int ThreadTileX = 8;
        const int ThreadTileY = 8;
        const int TILE_K = 16;
        testThreadTilingKernel(
            matmulsharedThreadTilingFloat4_kernel<ThreadDimx, ThreadDimy, ThreadTileX, ThreadTileY, TILE_K>, 
            d_A, d_B_row, d_C, M, K, P, 
            ThreadDimx, ThreadDimy, ThreadTileX, ThreadTileY, 
            "GPU Shared Memory Thread Tiling Float4 GEMM"
        );
        verify("GPU Shared Memory Thread Tiling Float4 GEMM");
    }

    testCublasGEMM(d_A, d_B_row, d_C, M, K, P);
    verify("cuBLAS Standard");

    cudaFree(d_A); cudaFree(d_B_row); cudaFree(d_B_col); cudaFree(d_C);
    free(h_A); free(h_B_row); free(h_B_col); free(h_C_ref); free(h_C_gpu);

    return 0;
}