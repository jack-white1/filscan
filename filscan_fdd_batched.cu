#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define ROTATION_SMEM_WIDTH 128
#define ROTATION_SMEM_HEIGHT 16

#define FRAGMENT_SUM_SMEM_WIDTH 16
#define FRAGMENT_SUM_SMEM_HEIGHT 16

struct header {
    const char *fileName;
    long fileSize;
    long headerSize;
    long dataSize;
    long nsamp;
    uint8_t nbits;
    uint16_t nchans;
    double tsamp;
    double fch1;
    double foff;
};

struct hostFilterbank{
    struct header header;
    uint8_t* data;
};

long get_file_size(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
}

long find_header_location(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    const char *search_str = "HEADER_END";
    int search_len = strlen(search_str);
    int match_len = 0;
    long byte_location = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                byte_location = ftell(file) - search_len;
                fclose(file);
                return byte_location;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return -1;
}

uint8_t find_nbits_value(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return UINT8_MAX;
    }

    const char *search_str = "nbits";
    int search_len = strlen(search_str);
    int match_len = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                uint8_t value;
                if (fread(&value, sizeof(uint8_t), 1, file) == 1) {
                    fclose(file);
                    return value;  // Assumes little-endian order
                }
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return UINT8_MAX;
}

uint16_t find_nchans_value(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return UINT16_MAX;
    }

    const char *search_str = "nchans";
    int search_len = strlen(search_str);
    int match_len = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                uint16_t value;
                if (fread(&value, sizeof(uint16_t), 1, file) == 1) {
                    fclose(file);
                    return value;  // Assumes little-endian order
                }
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return UINT16_MAX;
}

double find_tsamp_value(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1.0;
    }

    const char *search_str = "tsamp";
    int search_len = strlen(search_str);
    int match_len = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                double value;
                if (fread(&value, sizeof(double), 1, file) == 1) {
                    fclose(file);
                    return value;
                }
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return -1.0;
}

double find_fch1_value(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1.0;
    }

    const char *search_str = "fch1";
    int search_len = strlen(search_str);
    int match_len = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                double value;
                if (fread(&value, sizeof(double), 1, file) == 1) {
                    fclose(file);
                    return value;
                }
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return -1.0;
}

double find_foff_value(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1.0;
    }

    const char *search_str = "foff";
    int search_len = strlen(search_str);
    int match_len = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                double value;
                if (fread(&value, sizeof(double), 1, file) == 1) {
                    fclose(file);
                    return value;
                }
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fclose(file);
    return -1.0;
}

void readHeader(const char* filename, struct header* header){
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    const char *search_str = "HEADER_END";
    int search_len = strlen(search_str);
    int match_len = 0;
    long byte_location = 0;

    char ch;
    while (fread(&ch, 1, 1, file) == 1) {
        if (ch == search_str[match_len]) {
            match_len++;
            if (match_len == search_len) {
                byte_location = ftell(file) - search_len;
                break;
            }
        } else {
            match_len = 0;
        }
    }

    fseek(file, 0, SEEK_SET);
    header->fileName = filename;
    header->fileSize = get_file_size(filename);
    header->headerSize = byte_location;
    header->dataSize = header->fileSize - header->headerSize;
    header->nbits = find_nbits_value(filename);
    header->nchans = find_nchans_value(filename);
    header->tsamp = find_tsamp_value(filename);
    header->nsamp = header->dataSize / header->nchans;
    header->fch1 = find_fch1_value(filename);
    header->foff = find_foff_value(filename);

    fclose(file);
}

void printHeaderStruct(struct header* header){
    printf("File name:\t\t\t%s\n", header->fileName);
    printf("Total file size:\t\t%ld bytes\n", header->fileSize);
    printf("Header size:\t\t\t%ld bytes\n", header->headerSize);
    printf("Data size:\t\t\t%ld bytes\n", header->dataSize);
    printf("nbits:\t\t\t\t%d\n", header->nbits);
    printf("nchans:\t\t\t\t%d\n", header->nchans);
    printf("fch1:\t\t\t\t%lf\n", header->fch1);
    printf("foff:\t\t\t\t%lf\n", header->foff);
    printf("tsamp:\t\t\t\t%lf\n", header->tsamp);
    printf("nsamp:\t\t\t\t%ld\n", header->nsamp);
    printf("True observation time:\t\t%lf s\n", header->tsamp * header->nsamp);
}

void readFilterbankData(struct header* header, struct hostFilterbank* hostFilterbank){
    FILE *file = fopen(header->fileName, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    fseek(file, header->headerSize, SEEK_SET);
    fread(hostFilterbank->data, sizeof(uint8_t), header->dataSize, file);
    fclose(file);
}

__global__ void transpose_and_cast_uint8_t_to_padded_float(uint8_t* deviceData_uint8_t, float* deviceData_float, int nchans, int input_nsamps, int output_nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < input_nsamps && y < nchans) {
        deviceData_float[y * output_nsamps + x] = (float) deviceData_uint8_t[x * nchans + y];
    }
}

static __constant__ float cachedTimeShiftsPerDM[4096];

struct SharedMemory2DRotation {
    float2* data;
    __device__ float2* operator[](int idx) {
        return &data[idx * ROTATION_SMEM_WIDTH];
    }
};

__global__ void rotate_spectrum_smem(float2* deviceData_float2, float2* deviceData_output_float2, long nsamps, float FFTbinWidth, long nchans, float DMstart, float DMstep){
    //extern __shared__ float2 input[ROTATION_SMEM_HEIGHT][ROTATION_SMEM_WIDTH];   // ROTATION_SMEM_HEIGHT channels, ROTATION_SMEM_WIDTH samples per channel
    //extern __shared__ float2 output[ROTATION_SMEM_HEIGHT][ROTATION_SMEM_WIDTH];  // ROTATION_SMEM_HEIGHT DMs, ROTATION_SMEM_WIDTH samples per channel

    extern __shared__ float2 sharedMemory[];

    SharedMemory2DRotation input = { &sharedMemory[0] };
    SharedMemory2DRotation output = { &sharedMemory[ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH] };  // Offset by the size of the input array

    // threadIdx.x = 0 -> ROTATION_SMEM_WIDTH-1
    // threadIdx.y = 0
    
    // blockDim.x = ROTATION_SMEM_WIDTH
    // blockDim.y = 1

    // gridDim.x = nsamps / ROTATION_SMEM_WIDTH = ((nextPowerOf2/2)+1) / ROTATION_SMEM_WIDTH
    // gridDim.y = nchans / ROTATION_SMEM_HEIGHT

    // starting channel = blockIdx.y * ROTATION_SMEM_HEIGHT

    // copy channel 0 -> ROTATION_SMEM_HEIGHT - 1 data to shared memory, ROTATION_SMEM_WIDTH samples wide
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * ROTATION_SMEM_HEIGHT;

    if (global_x < nsamps && global_y < nchans){
        for (int y_i = 0; y_i < ROTATION_SMEM_HEIGHT; y_i++){
            input[y_i][threadIdx.x] = deviceData_float2[(global_y + y_i) * nsamps + global_x];       
        }
    }

    // set output to 0
    for (int y_i = 0; y_i < ROTATION_SMEM_HEIGHT; y_i++){
        output[y_i][threadIdx.x].x = 0.0f;
        output[y_i][threadIdx.x].y = 0.0f;
    }

    __syncthreads();

    float multiplier = 2.0 * M_PI * FFTbinWidth * global_x;
    float DM;

    // initialise rotations
    // rotate channel 0 -> ROTATION_SMEM_HEIGHT - 1 data in shared memory, ROTATION_SMEM_WIDTH samples wide
    for (int DM_i = 0; DM_i < ROTATION_SMEM_HEIGHT; DM_i++){
        DM = DMstart + DM_i * DMstep;
        if (global_x < nsamps && global_y < nchans){
            float2 value;
            float phase;
            float s, c;
            for (int y_i = 0; y_i < ROTATION_SMEM_HEIGHT; y_i++){
                value = input[y_i][threadIdx.x];
                phase = multiplier * cachedTimeShiftsPerDM[global_y + y_i] * DM;
                __sincosf(phase, &s, &c);
                input[y_i][threadIdx.x].x += value.x * c - value.y * s;
                input[y_i][threadIdx.x].y += value.x * s + value.y * c;
                output[DM_i][threadIdx.x].x += input[y_i][threadIdx.x].x;
                output[DM_i][threadIdx.x].y += input[y_i][threadIdx.x].y;
            }
        }
    }

    __syncthreads();

    // copy channel 0 -> ROTATION_SMEM_HEIGHT - 1 data from shared memory to global memory, ROTATION_SMEM_WIDTH samples wide
    if (global_x < nsamps && global_y < nchans){
        for (int y_i = 0; y_i < ROTATION_SMEM_HEIGHT; y_i++){
            deviceData_output_float2[(global_y + y_i) * nsamps + global_x] = output[y_i][threadIdx.x];  // overreaching?
        }
    }
}



__global__ void reassemble_fragments(float2* deviceData_float2_dedispersed_fragments, float2* deviceData_float2_dedispersed, int num_ddtr_fragments_y, long ddtr_fragment_num_float2s, int localDMOffset, int globalDMOffset, int fragment_dim_y, long nsamps){
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;

    float2 value;
    value.x = 0.0f;
    value.y = 0.0f;

    for (int frag_y_idx = 0; frag_y_idx < num_ddtr_fragments_y; frag_y_idx++){
        value.x += deviceData_float2_dedispersed_fragments[(frag_y_idx * nsamps * fragment_dim_y) + global_x].x;
        value.y += deviceData_float2_dedispersed_fragments[(frag_y_idx * nsamps * fragment_dim_y) + global_x].y;
    }

    deviceData_float2_dedispersed[(globalDMOffset + localDMOffset) * nsamps + global_x] = value;

}

void compute_time_shifts(float* timeShifts, float f1, float foff, int nchans, float DM) {
    for (int i = 0; i < nchans; i++) {
        float f2 = f1 + foff * i;

        // convert to GHz
        float f1_GHz = f1 / 1000.0;
        float f2_GHz = f2 / 1000.0;
        float k = 4.148808;

        // compute the time shift in ms
        float timeShift_ms = k * DM * (1.0 / (f1_GHz * f1_GHz) - 1.0 / (f2_GHz * f2_GHz));

        // convert to seconds
        timeShifts[i] = - timeShift_ms / 1000.0;
    }
}


const char* filscan_frame = 

"   ______________ __                    \n"
"    _____  ____(_) /_____________ _____ \n"
"     ___  /_  / / / ___/ ___/ __ `/ __ \\ \n"
"      _  __/ / / (__  ) /__/ /_/ / / / /\n"
"      /_/   /_/_/____/\\___/\\__,_/_/ /_/\n\n";



int main(int argc, char *argv[]) {
    printf("%s", filscan_frame);

    if (argc != 2) {
        printf("Usage: %s <file_name>\n", argv[0]);
        return 1;
    }

    // initialise error, timing and available memory variables for use throughout the program
    cudaError_t error = cudaGetLastError();
    struct timeval start, end;
    size_t availableMemory, totalMemory;
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory:\t\t%ld MB\n", availableMemory / 1024 / 1024);
    printf("Total memory:\t\t\t%ld MB\n", totalMemory / 1024 / 1024);




    struct header header;
    readHeader(argv[1], &header);
    printHeaderStruct(&header);

    struct hostFilterbank hostFilterbank;
    hostFilterbank.header = header;
    hostFilterbank.data = (uint8_t*) malloc(header.dataSize * sizeof(uint8_t));
    readFilterbankData(&header, &hostFilterbank);

    long nsamps = (long) header.nsamp;
    long nextPowerOf2 = 1;
    while (nextPowerOf2 < nsamps) {
        nextPowerOf2 *= 2;
    }
    printf("Next power of 2:\t\t%ld\n", nextPowerOf2);
    printf("Padded observation time:\t%lf\n", header.tsamp * nextPowerOf2);
    printf("FFT bin width\t\t\t%lf Hz\n", 1.0 / (header.tsamp * nextPowerOf2));

    float FFTbinWidth = 1.0 / (header.tsamp * nextPowerOf2);

    long nchans = (long) header.nchans;

    printf("Data length:\t\t\t%ld bytes\n", nchans * nextPowerOf2);

    // allocate memory on the device
    u_int8_t* deviceData_uint8_t;
    float* deviceData_float;
    float2* deviceData_float2;

    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));
    cudaMalloc((void**)&deviceData_float, nchans * nextPowerOf2 * sizeof(float));
    cudaMalloc((void**)&deviceData_float2, ((nextPowerOf2/2)+1) * nchans * sizeof(float2));
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after first mallocs:\t\t%ld MB\n", availableMemory / 1024 / 1024);

    cudaMemset(deviceData_float, 0, nchans * nextPowerOf2 * sizeof(float));

    cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    // transpose and cast
    dim3 dimBlock(32, 32);
    dim3 dimGrid((nsamps + dimBlock.x - 1) / dimBlock.x, (nchans + dimBlock.y - 1) / dimBlock.y);
    transpose_and_cast_uint8_t_to_padded_float<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_float, nchans, nsamps, nextPowerOf2);
    cudaDeviceSynchronize();
    cudaFree(deviceData_uint8_t);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free uint8:\t\t%ld MB\n", availableMemory / 1024 / 1024);


    // cufft each channel, storing the output in the float2 array
    cufftHandle plan;
    cufftPlan1d(&plan, nextPowerOf2, CUFFT_R2C, nchans);
    cufftExecR2C(plan, deviceData_float, deviceData_float2);
    cudaDeviceSynchronize();
    cudaFree(deviceData_float);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free float:\t\t%ld MB\n", availableMemory / 1024 / 1024);


    // compute the time shifts for each channel
    float* timeShifts = (float*) malloc(nchans * sizeof(float));
    compute_time_shifts(timeShifts, header.fch1, header.foff, nchans, 1.0);
    cudaMemcpyToSymbol(cachedTimeShiftsPerDM, timeShifts, nchans * sizeof(float));
    

    int num_ddtr_fragments = nchans / ROTATION_SMEM_HEIGHT;
    long ddtr_fragment_num_float2s = ((nextPowerOf2/2)+1) * ROTATION_SMEM_HEIGHT;

    // cudaMalloc the output fragments array
    float2* deviceData_float2_dedispersed_fragments;
    cudaMalloc((void**)&deviceData_float2_dedispersed_fragments, num_ddtr_fragments * ddtr_fragment_num_float2s * sizeof(float2));

    //printf("Tried to malloc %ld MB\n", num_ddtr_fragments * ddtr_fragment_num_float2s * sizeof(float2) / 1024 / 1024);
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after fragments malloc:\t\t%ld MB\n", availableMemory / 1024 / 1024);

    // rotate the spectrum channelwise
    float startDM = 50;
    float DMstep = 0.1;
    dim3 dimBlock2(ROTATION_SMEM_WIDTH, 1);
    dim3 dimGrid2(((nextPowerOf2/2)+1 + dimBlock2.x - 1) / dimBlock2.x, nchans / ROTATION_SMEM_HEIGHT);

    cudaFuncSetAttribute(rotate_spectrum_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH * sizeof(float2));
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 0: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    rotate_spectrum_smem<<<dimGrid2, dimBlock2, 2 * ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH * sizeof(float2)>>>(deviceData_float2, deviceData_float2_dedispersed_fragments, ((nextPowerOf2/2)+1), FFTbinWidth, nchans, startDM, DMstep);
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 1: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // reassemble the fragments
    float2* deviceData_float2_dedispersed;
    cudaMalloc((void**)&deviceData_float2_dedispersed, ((nextPowerOf2/2)+1) * ROTATION_SMEM_HEIGHT * sizeof(float2));

    int num_ddtr_fragments_y = nchans / ROTATION_SMEM_HEIGHT;
    printf("num_ddtr_fragments_y = %d\n", num_ddtr_fragments_y);
    int localDMOffset = 0;
    int globalDMOffset = 0;
    int fragment_dim_y = ROTATION_SMEM_HEIGHT;
    dim3 dimBlock3(FRAGMENT_SUM_SMEM_WIDTH, 1);
    dim3 dimGrid3(((nextPowerOf2/2)+1 + dimBlock3.x - 1) / dimBlock3.x, 1);



    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 2: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    // start timer
    gettimeofday(&start, NULL);
    for (int DM_i = 0; DM_i < 16; DM_i++){


        reassemble_fragments<<<dimGrid3, dimBlock3, (num_ddtr_fragments + 1) * FRAGMENT_SUM_SMEM_WIDTH * sizeof(float2)>>>(deviceData_float2_dedispersed_fragments, deviceData_float2_dedispersed, num_ddtr_fragments_y, ddtr_fragment_num_float2s, localDMOffset, globalDMOffset, fragment_dim_y, ((nextPowerOf2/2)+1));
        cudaDeviceSynchronize();

        // check cuda error

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error 3: %s\n", cudaGetErrorString(error));
            return 1;
        }
        localDMOffset += 1;
        globalDMOffset += 1;

            // stop timer
    gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    printf("Time taken: %lf s\n", time_taken);
    }


    // free memory
    cudaFree(deviceData_float2);
    free(hostFilterbank.data);
    free(timeShifts);



    return 0;
}
