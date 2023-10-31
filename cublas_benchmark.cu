#include <cublas_v2.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error in " << __FILE__ << " at line " << __LINE__ << ": " << stat; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

    int main() {
        const int N = 4096; // size of the matrix, adjust as needed
        float *A, *B, *C;
        half *A16, *B16, *C16;
    
        cublasHandle_t handle; // cuBLAS context
        CHECK_CUBLAS(cublasCreate(&handle)); // Initialize cuBLAS context
    
        CHECK_CUDA(cudaMalloc(&A, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&B, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&C, N * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&A16, N * N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&B16, N * N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&C16, N * N * sizeof(half)));
    
        // Initialize matrices A and B here...
    
        const float alpha = 1.0f; // Moved these before their first usage
        const float beta = 0.0f;
    
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &alpha, A, N, &beta, B, N, C, N));
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A16, CUDA_R_16F, N, B16, CUDA_R_16F, N, &beta, C16, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    
        // Record the start time
        auto start = std::chrono::high_resolution_clock::now();
    
        // Do the matrix multiplication
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A16, CUDA_R_16F, N, B16, CUDA_R_16F, N, &beta, C16, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel completion before stopping the timer
    
        // Record the end time
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        // Calculate throughput in GB/s
        double totalDataGB = 3.0 * N * N * sizeof(half) / (1024.0 * 1024.0 * 1024.0);  // 3 matrices * N * N elements * 2 bytes each
        double throughputGBs = totalDataGB / elapsed.count();

        // Calculate throughput in TFLOPs/s
        double totalFLOPs = (2.0 * N * N * N - N * N) / 1e12;  // 2N^3 - N^2 FLOPs
        double throughputTFLOPs = totalFLOPs / elapsed.count();

        std::cout << "Matrix multiplication took: " << elapsed.count() << " seconds." << std::endl;
        std::cout << "Throughput: " << throughputGBs << " GB/s" << std::endl;
        std::cout << "Performance: " << throughputTFLOPs << " TFLOPs/s" << std::endl;

    
        // Clean up
        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C));
        CHECK_CUDA(cudaFree(A16));
        CHECK_CUDA(cudaFree(B16));
        CHECK_CUDA(cudaFree(C16));
    
        CHECK_CUBLAS(cublasDestroy(handle)); // Destroy cuBLAS context
    
        return 0;
    }
