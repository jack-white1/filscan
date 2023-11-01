#include <iostream>
#include <mma.h>

using namespace nvcuda;

const int M = 16;
const int N = 16;
const int K = 16;

__global__ void wmma_test_kernel() {
    // Define the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, half> c_frag;

    // Fill the fragments with dummy values
    wmma::fill_fragment(a_frag, __float2half(1.0f));
    wmma::fill_fragment(b_frag, __float2half(1.0f));
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // No output, just to check compilation
}

int main() {
    wmma_test_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    std::cout << "WMMA test completed!" << std::endl;
    return 0;
}
