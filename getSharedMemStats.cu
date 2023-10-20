#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, device) == cudaSuccess) {
            printf("Properties for device %d:\n", device);
            printf("=====================================\n");
            printf("Name: %s\n", deviceProp.name);
            printf("Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
            printf("Shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
            printf("Registers per block: %d\n", deviceProp.regsPerBlock);
            printf("Warp size: %d\n", deviceProp.warpSize);
            printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
            printf("Max thread dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
            printf("Max grid dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
            printf("Clock rate: %d kHz\n", deviceProp.clockRate);
            printf("Total constant memory: %zu bytes\n", deviceProp.totalConstMem);
            printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
            printf("Texture alignment: %zu bytes\n", deviceProp.textureAlignment);
            printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
            printf("Kernel execution timeout: %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
            printf("Integrated GPU sharing Host Memory: %s\n", deviceProp.integrated ? "Yes" : "No");
            printf("Can map host memory: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
            printf("Compute mode: %d\n", deviceProp.computeMode);
            printf("Max texture 1D dimensions: %d\n", deviceProp.maxTexture1D);
            printf("Max texture 2D dimensions: (%d, %d)\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
            printf("Max texture 3D dimensions: (%d, %d, %d)\n", deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
            printf("Max texture 1D layered dimensions: (%d x %d layers)\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
            printf("Max texture 2D layered dimensions: (%d x %d layers)\n", deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1]);
            printf("Surface alignment: %zu bytes\n", deviceProp.surfaceAlignment);
            printf("Concurrent kernel execution: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
            printf("ECC support: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
            printf("PCI bus ID: %d\n", deviceProp.pciBusID);
            printf("PCI device ID: %d\n", deviceProp.pciDeviceID);
            printf("TCC driver mode: %s\n", deviceProp.tccDriver ? "Yes" : "No");
            printf("Unified addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
            printf("Max pitch allowed for memory copies: %zu bytes\n", deviceProp.memPitch);
            printf("Peak memory clock frequency: %d kHz\n", deviceProp.memoryClockRate);
            printf("Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
            printf("L2 cache size: %d bytes\n", deviceProp.l2CacheSize);
            printf("Maximum 1D layered texture width and layers: %d x %d layers\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
            printf("Maximum 2D layered texture dimensions and layers: (%d x %d) x %d layers\n", deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
            printf("Surface alignment: %zu bytes\n", deviceProp.surfaceAlignment);
            printf("Concurrent copy and execution: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
            printf("Number of asynchronous engines: %d\n", deviceProp.asyncEngineCount);
            printf("Page-locked host memory mapping: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
            printf("Compute mode: %d (0: Default, 1: Exclusive, 2: Prohibited, 3: Exclusive Process)\n", deviceProp.computeMode);
            printf("Direct managed memory access from PCI Bus: %s\n", deviceProp.directManagedMemAccessFromHost ? "Yes" : "No");
            // ... Additional properties can be added here as required

            printf("\n");
        }
    }

    return 0;
}
