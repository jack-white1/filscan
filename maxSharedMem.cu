#include <cuda_runtime.h>
#include <stdio.h>

__global__ void my_kernel() {
    extern __shared__ float sharedMem[];
    // Do something with shared memory if needed
}

int main() {

    // Set the maximum dynamic shared memory size for the kernel
    size_t desiredSharedMemSize = 99 * 1024;  // 96KB as an example
    cudaError_t err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set the attribute: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, desiredSharedMemSize);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set the attribute: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Now, let's retrieve the set value to verify
    cudaFuncAttributes attributes;
    err = cudaFuncGetAttributes(&attributes, my_kernel);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get function attributes: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum dynamic shared memory size for 'my_kernel': %d bytes\n", attributes.maxDynamicSharedSizeBytes);

    return 0;
}
