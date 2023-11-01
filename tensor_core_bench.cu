#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

const int M = 16;
const int N = 16;
const int K = 16;

const int WARP_SIZE = 32;
const int NUM_BLOCKS = 1024;
const int NUM_WARPS_PER_BLOCK = 4;
const int NUM_ITERATIONS = 4;

__global__ void wmmaKernel(half *A, half *B, half *C, int num_iterations) {
    // Define the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Load the matrix from global memory
    wmma::load_matrix_sync(a_frag, A + (blockIdx.x * M * K), K);
    wmma::load_matrix_sync(b_frag, B + (blockIdx.x * K * N), N);

    clock_t start, stop;
    start = clock64();

    // Perform matrix multiplication
    #pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();

    stop = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0)
    printf("Elapsed clock cycles: %d\n", (int)(stop - start));

    // Store the result back to global memory
    wmma::store_matrix_sync(C + (blockIdx.x * M * N), c_frag, N, wmma::mem_row_major);
}

int main() {
    // Allocate and initialize matrices on the host
    half *h_A = new half[M * K * NUM_BLOCKS];
    half *h_B = new half[K * N * NUM_BLOCKS];
    half *h_C = new half[M * N * NUM_BLOCKS];

    for (int i = 0; i < M * K * NUM_BLOCKS; i++) h_A[i] = __float2half(rand() % 100);
    for (int i = 0; i < K * N * NUM_BLOCKS; i++) h_B[i] = __float2half(rand() % 100);

    // Allocate matrices on the device
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * NUM_BLOCKS * sizeof(half));
    cudaMalloc(&d_B, K * N * NUM_BLOCKS * sizeof(half));
    cudaMalloc(&d_C, M * N * NUM_BLOCKS * sizeof(half));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, M * K * NUM_BLOCKS * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * NUM_BLOCKS * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    wmmaKernel<<<NUM_BLOCKS, WARP_SIZE * NUM_WARPS_PER_BLOCK>>>(d_A, d_B, d_C, NUM_ITERATIONS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
