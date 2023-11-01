#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <mma.h>

#define ROTATE_BLOCK_DIM_X 16
#define ROTATE_BLOCK_DIM_Y 16

using namespace nvcuda;

struct header {
    const char *fileName;
    long fileSize;
    long headerSize;
    long dataSize;
    long nsamp;
    long paddedLength;
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

    long nextPowerOf2 = 1;
    while (nextPowerOf2 < header->nsamp) {
        nextPowerOf2 *= 2;
    }

    header->paddedLength = nextPowerOf2;

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

__global__ void rotate_spectrum(float2* inputArray, float2* outputArray, long nchans, long nsamps, float DM){
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y;
    long outputIndex = y * nsamps + x;

    if (x < nsamps && y < nchans) {
        float phase = x * DM * cachedTimeShiftsPerDM[y];
        float2 input = inputArray[outputIndex];
        float2 output;
        float s, c;
        sincosf(phase, &s, &c);
        output.x = input.x * c - input.y * s;
        output.y = input.x * s + input.y * c;
        outputArray[outputIndex] = output;
    }
}

__global__ void sum_across_channels(float2* inputArray, float2* outputArray, long nchans, long nsamps){
    long x = blockIdx.x * blockDim.x + threadIdx.x;

    float2 sum;
    sum.x = 0.0;
    sum.y = 0.0;

    if (x < nsamps) {
        for (long y = 0; y < nchans; y++) {
            float2 value = inputArray[y * nsamps + x];
            sum.x += value.x;
            sum.y += value.y;
        }
        outputArray[x] = sum;
    }
}


__global__ void rotate_spectrum_smem_32_square(
    float2* inputData, float2* outputData, long nsamps, long nchans, float DMstart, float DMstep
) {
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Define constants for thread block dimensions
    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 32;

    __shared__ float2 input[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float2 intermediate[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float2 output[BLOCK_DIM_Y][BLOCK_DIM_X];

    // Load data from global memory to shared memory
    if (global_x < nsamps && global_y < nchans) {
        input[local_y][local_x] = inputData[global_y * nsamps + global_x];
        output[local_y][local_x].x = 0.0f;
        output[local_y][local_x].y = 0.0f;
    }
    __syncthreads();

    float DM = DMstart;
    for (int DM_idx = 0; DM_idx < BLOCK_DIM_Y; DM_idx++) {
        float phase = global_x * DM * cachedTimeShiftsPerDM[global_y];
        float2 input_value = input[local_y][local_x];
        float s, c;
        sincosf(phase, &s, &c);
        intermediate[local_y][local_x].x = input_value.x * c - input_value.y * s;
        intermediate[local_y][local_x].y = input_value.x * s + input_value.y * c;

        __syncthreads();

        // Hierarchical reduction with loop unrolling
        if (local_y < 16) {
            float2 sum = intermediate[local_y][local_x];
            sum.x += intermediate[local_y + 16][local_x].x;
            sum.y += intermediate[local_y + 16][local_x].y;
            intermediate[local_y][local_x] = sum;
        }
        __syncthreads();

        if (local_y < 8) {
            float2 sum = intermediate[local_y][local_x];
            sum.x += intermediate[local_y + 8][local_x].x;
            sum.y += intermediate[local_y + 8][local_x].y;
            intermediate[local_y][local_x] = sum;
        }
        __syncthreads();

        // Warp-pruned reduction for the last 8 rows
        if (local_y < 4) {
            float2 sum = intermediate[local_y][local_x];
            sum.x += intermediate[local_y + 4][local_x].x;
            sum.y += intermediate[local_y + 4][local_x].y;
            sum.x += intermediate[local_y + 2][local_x].x;
            sum.y += intermediate[local_y + 2][local_x].y;
            sum.x += intermediate[local_y + 1][local_x].x;
            sum.y += intermediate[local_y + 1][local_x].y;
            intermediate[local_y][local_x] = sum;
        }
        __syncthreads();

        // Write to output shared memory array
        if (local_y == 0) {
            output[DM_idx][local_x] = intermediate[0][local_x];
        }

        DM += DMstep;
    }
    __syncthreads();

    // Copy data from shared memory to global memory
    if (global_x < nsamps && global_y < nchans) {
        outputData[global_y * nsamps + global_x] = output[local_y][local_x];
    }
    __syncthreads();
}

__global__ void tensor_core_rotate_spectrum_smem_32_square(float2* inputData, float2* outputData, long nsamps, long nchans, float DMstart, float DMstep) {
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int first_thread_in_block_x = blockIdx.x * blockDim.x;

    __shared__ float2 output[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    __shared__ float2 phase[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    __shared__ float timeShiftsArray[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    
    __shared__ half inputReal_half[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    __shared__ half inputImag_half[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    __shared__ half phasorReal_half[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];
    __shared__ half phasorImag_half[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];

    if (global_x < nsamps && global_y < nchans) {
        inputReal_half[local_y][local_x] = __float2half(inputData[global_y * nsamps + global_x].x);
        inputImag_half[local_y][local_x] = __float2half(inputData[global_y * nsamps + global_x].y);
    }
    
    output[local_y][local_x].x = 0.0f;
    output[local_y][local_x].y = 0.0f;

    if (local_x == 0 && local_y == 0) {
        for(int i = 0; i < ROTATE_BLOCK_DIM_Y; i++) {
            for(int j = 0; j < ROTATE_BLOCK_DIM_X; j++) {
                timeShiftsArray[i][j] = first_thread_in_block_x * (DMstart + j * DMstep) * cachedTimeShiftsPerDM[global_y + i];
            }
        }
    }

    __syncthreads();

    float s, c;
    sincosf(timeShiftsArray[local_y][local_x], &s, &c);
    phasorReal_half[local_y][local_x] = __float2half(c);
    phasorImag_half[local_y][local_x] = __float2half(-s);

    const int M = ROTATE_BLOCK_DIM_Y;
    const int N = ROTATE_BLOCK_DIM_X;
    const int K = ROTATE_BLOCK_DIM_X;

    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> input_real_frag, input_imag_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> phasor_real_frag, phasor_imag_frag;
    wmma::fragment<wmma::accumulator, M, N, K, half> output_real_frag, output_imag_frag;

    wmma::load_matrix_sync(input_real_frag, &inputReal_half[0][0], K);
    wmma::load_matrix_sync(input_imag_frag, &inputImag_half[0][0], K);
    wmma::load_matrix_sync(phasor_real_frag, &phasorReal_half[0][0], N);
    wmma::load_matrix_sync(phasor_imag_frag, &phasorImag_half[0][0], N);

    wmma::fill_fragment(output_real_frag, 0.0f);
    wmma::fill_fragment(output_imag_frag, 0.0f);

    wmma::mma_sync(output_real_frag, input_real_frag, phasor_real_frag, output_real_frag);
    wmma::mma_sync(output_real_frag, input_imag_frag, phasor_imag_frag, output_real_frag);
    wmma::mma_sync(output_imag_frag, input_real_frag, phasor_imag_frag, output_imag_frag);
    wmma::mma_sync(output_imag_frag, input_imag_frag, phasor_real_frag, output_imag_frag);

    __shared__ half result_real[M * N];
    __shared__ half result_imag[M * N];
    wmma::store_matrix_sync(result_real, output_real_frag, N, wmma::mem_row_major);
    wmma::store_matrix_sync(result_imag, output_imag_frag, N, wmma::mem_row_major);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            output[i][j].x = __half2float(result_real[i * N + j]);
            output[i][j].y = __half2float(result_imag[i * N + j]);
        }
    }

    if (global_x < nsamps && global_y < nchans) {
        outputData[global_y * nsamps + global_x] = output[local_y][local_x];
    }
}


__global__ void unoptimised_rotate_spectrum_smem_32_square(float2* inputData, float2* outputData, long nsamps, long nchans, float DMstart, float DMstep){
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float2 output[ROTATE_BLOCK_DIM_Y][ROTATE_BLOCK_DIM_X];

    float2 inputDataPoint;
    if (global_x < nsamps && global_y < nchans){
        inputDataPoint = inputData[global_y * nsamps + global_x];
    }
    
    // set output to 0
    output[local_y][local_x].x = 0.0f;
    output[local_y][local_x].y = 0.0f;

    float cachedTimeShiftPerDM = cachedTimeShiftsPerDM[global_y];
    float2 intermediate_value;
    float DM = DMstart;
    float phase;
    for (int DM_idx = 0; DM_idx < ROTATE_BLOCK_DIM_Y; DM_idx++){
        phase = global_x * DM * cachedTimeShiftPerDM;
        if (global_x < 10 && global_y == 10){
            printf("global_x: %d, phase: %f, DM: %f\n", global_x, phase, DM);
        }
        float s, c;
        sincosf(phase, &s, &c); 
        intermediate_value.x = fmaf(inputDataPoint.x, c, -inputDataPoint.y * s);
        intermediate_value.y = fmaf(inputDataPoint.x, s, inputDataPoint.y * c);
        __syncthreads();
        
        // Start the hierarchical reduction across the y-axis using warp-level primitives
        unsigned mask = __ballot_sync(0xFFFFFFFF, 1);
       
        // Loop unrolling for strides: 16, 8, 4, 2, and 1
        intermediate_value.x += __shfl_down_sync(mask, intermediate_value.x, 16);
        intermediate_value.y += __shfl_down_sync(mask, intermediate_value.y, 16);

        intermediate_value.x += __shfl_down_sync(mask, intermediate_value.x, 8);
        intermediate_value.y += __shfl_down_sync(mask, intermediate_value.y, 8);

        intermediate_value.x += __shfl_down_sync(mask, intermediate_value.x, 4);
        intermediate_value.y += __shfl_down_sync(mask, intermediate_value.y, 4);

        intermediate_value.x += __shfl_down_sync(mask, intermediate_value.x, 2);
        intermediate_value.y += __shfl_down_sync(mask, intermediate_value.y, 2);

        intermediate_value.x += __shfl_down_sync(mask, intermediate_value.x, 1);
        intermediate_value.y += __shfl_down_sync(mask, intermediate_value.y, 1);

        // Write the final reduced value to output shared memory array
        if (local_y == 0){
            output[DM_idx][local_x] = intermediate_value;
        }

        DM += DMstep;
    }

    __syncthreads();
    // copy data from shared memory to global memory
    if (global_x < nsamps && global_y < nchans){
        outputData[global_y * nsamps + global_x] = output[local_y][local_x];
    }
}

// 4096 channels means 128 blocks of 32 DMs, this kernel should take the Nth DM from each block and sum them
__global__ void sum_across_channels_smem(float2* inputData, float2* outputData, long nsamps, long nchans){
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int DM_idx = blockIdx.y;

    // blockDim.x = 8
    // blockDim.y = 128

    __shared__ float2 input[128][8];

    // copy data from global memory to shared memory
    if (global_x < nsamps && local_y * 32 < nchans){
        input[local_y][local_x] = inputData[(local_y * 32 + DM_idx) * nsamps + global_x];
    }

    // parallel reduction sum across the y axis of input
    for (int stride = 4; stride > 0; stride /= 2) {
        if (local_x < stride) {
            input[local_y][local_x].x += input[local_y][local_x + stride].x;
            input[local_y][local_x].y += input[local_y][local_x + stride].y;
        }
        __syncthreads();
    }

    // write first row of input to output
    outputData[DM_idx * nsamps + global_x] = input[0][local_x];
}




void compute_time_shifts(float* timeShifts, float f1, float foff, int nchans, float DM, float FFTbinWidth) {
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
        timeShifts[i] *= 2.0 * M_PI * FFTbinWidth;
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
    gettimeofday(&start, NULL);
    size_t availableMemory, totalMemory;

    // print the available memory on the GPU
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

    printf("Next power of 2:\t\t%ld\n", header.paddedLength);
    printf("Padded observation time:\t%lf\n", header.tsamp * header.paddedLength);
    printf("FFT bin width\t\t\t%lf Hz\n", 1.0 / (header.tsamp * header.paddedLength));

    float FFTbinWidth = 1.0 / (header.tsamp * header.paddedLength);

    printf("Data length:\t\t\t%ld bytes\n", header.nchans * header.paddedLength);

    // allocate memory on the device
    u_int8_t* deviceData_uint8_t;
    float* deviceData_float;
    float2* deviceData_float2_raw;
    float2* deviceData_float2_dedispersed;
    float2* deviceData_float2_single_spectrum;

    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));
    cudaMalloc((void**)&deviceData_float, header.nchans * header.paddedLength * sizeof(float));
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("\nAvailable memory after first mallocs:\t\t%ld MB\n", availableMemory / 1024 / 1024);



    cudaMemset(deviceData_float, 0, header.nchans * header.paddedLength * sizeof(float));

    cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    // transpose and cast
    dim3 dimBlock(32, 32);
    dim3 dimGrid((header.paddedLength + dimBlock.x - 1) / dimBlock.x, (header.nchans + dimBlock.y - 1) / dimBlock.y);
    transpose_and_cast_uint8_t_to_padded_float<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_float, header.nchans, header.nsamp, header.paddedLength);
    cudaDeviceSynchronize();
    cudaFree(deviceData_uint8_t);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free uint8:\t\t%ld MB\n", availableMemory / 1024 / 1024);


    cudaMalloc((void**)&deviceData_float2_raw, ((header.paddedLength/2)+1) * header.nchans * sizeof(float2));
    // cufft each channel, storing the output in the float2 array
    cufftHandle plan;
    cufftPlan1d(&plan, header.paddedLength, CUFFT_R2C, header.nchans);
    cufftExecR2C(plan, deviceData_float, deviceData_float2_raw);
    cudaDeviceSynchronize();
    cudaFree(deviceData_float);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free float:\t\t%ld MB\n", availableMemory / 1024 / 1024);
    cudaMalloc((void**)&deviceData_float2_dedispersed, ((header.paddedLength/2)+1) * header.nchans * sizeof(float2));
    cudaMalloc((void**)&deviceData_float2_single_spectrum, ((header.paddedLength/2)+1) * sizeof(float2));




    // compute the time shifts for each channel
    float* timeShifts = (float*) malloc(header.nchans * sizeof(float));
    compute_time_shifts(timeShifts, header.fch1, header.foff, header.nchans, 1.0, FFTbinWidth);
    cudaMemcpyToSymbol(cachedTimeShiftsPerDM, timeShifts, header.nchans * sizeof(float));

    float* deviceTimeShifts;
    cudaMalloc((void**)&deviceTimeShifts, header.nchans * sizeof(float));
    cudaMemcpy(deviceTimeShifts, timeShifts, header.nchans * sizeof(float), cudaMemcpyHostToDevice);

    // get last cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 1: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float2* deviceData_float2_dedispersed_block;
    cudaMalloc((void**)&deviceData_float2_dedispersed_block, ((header.paddedLength/2)+1) * 32 * sizeof(float2));

    float DM = 49;
    float DM_step = 1;

    //for (int DM_idx = 0; DM_idx < 8; DM_idx++){
        DM += DM_step;

        // time the kernel
        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventRecord(startKernel, 0);

        // rotate the spectrum
        dim3 dimBlockRotation(1024, 1);
        dim3 dimGridRotation(((header.paddedLength/2)+1 + dimBlockRotation.x - 1) / dimBlockRotation.x, header.nchans);
        rotate_spectrum<<<dimGridRotation, dimBlockRotation>>>(deviceData_float2_raw, deviceData_float2_dedispersed, (long)header.nchans, (header.paddedLength/2) + 1, DM);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel, 0);
        cudaEventSynchronize(stopKernel);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startKernel, stopKernel);
        printf("\nRotation kernel (naive implementation, individual DM) time:\t%lf s\n", elapsedTime / 1000.0);


        // time the kernel
        cudaEvent_t startKernel_smem, stopKernel_smem;
        cudaEventCreate(&startKernel_smem);
        cudaEventCreate(&stopKernel_smem);
        cudaEventRecord(startKernel_smem, 0);

        // rotate the spectrum using smem version
        dim3 dimBlockRotation_smem(ROTATE_BLOCK_DIM_X, ROTATE_BLOCK_DIM_Y);
        //dim3 dimBlockRotation_smem(32, 32); 
        dim3 dimGridRotation_smem((header.paddedLength + dimBlockRotation_smem.x - 1) / dimBlockRotation_smem.x, (header.nchans + dimBlockRotation_smem.y - 1) / dimBlockRotation_smem.y);
        unoptimised_rotate_spectrum_smem_32_square<<<dimGridRotation_smem, dimBlockRotation_smem>>>(deviceData_float2_raw, deviceData_float2_dedispersed, (header.paddedLength/2) + 1, header.nchans, DM, DM_step);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel_smem, 0);
        cudaEventSynchronize(stopKernel_smem);
        float elapsedTime_smem;
        cudaEventElapsedTime(&elapsedTime_smem, startKernel_smem, stopKernel_smem);
        printf("Rotation kernel (smem implementation, %d DMs) time:\t\t%lf s\n", ROTATE_BLOCK_DIM_Y, elapsedTime_smem / 1000.0);

        // check cuda error
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error 2: %s\n", cudaGetErrorString(error));
            return 1;
        }

        // begin timer
        cudaEvent_t startKernel_tensor_core, stopKernel_tensor_core;
        cudaEventCreate(&startKernel_tensor_core);
        cudaEventCreate(&stopKernel_tensor_core);
        cudaEventRecord(startKernel_tensor_core, 0);


        // use tensor core kernel to rotate the spectrum
        dim3 dimBlockRotation_tensor_core(ROTATE_BLOCK_DIM_X, ROTATE_BLOCK_DIM_Y);
        dim3 dimGridRotation_tensor_core((header.paddedLength + dimBlockRotation_tensor_core.x - 1) / dimBlockRotation_tensor_core.x, (header.nchans + dimBlockRotation_tensor_core.y - 1) / dimBlockRotation_tensor_core.y);
        tensor_core_rotate_spectrum_smem_32_square<<<dimGridRotation_tensor_core, dimBlockRotation_tensor_core>>>(deviceData_float2_raw, deviceData_float2_dedispersed, (header.paddedLength/2) + 1, header.nchans, DM, DM_step);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel_tensor_core, 0);
        cudaEventSynchronize(stopKernel_tensor_core);
        float elapsedTime_tensor_core;
        cudaEventElapsedTime(&elapsedTime_tensor_core, startKernel_tensor_core, stopKernel_tensor_core);
        printf("Rotation kernel (tensor core implementation, %d DMs) time:\t%lf s\n", ROTATE_BLOCK_DIM_Y, elapsedTime_tensor_core / 1000.0);







        // time the kernel
        cudaEvent_t startKernel2, stopKernel2;
        cudaEventCreate(&startKernel2);
        cudaEventCreate(&stopKernel2);
        cudaEventRecord(startKernel2, 0);

        // sum across channels
        dim3 dimBlockSum(1024, 1);
        dim3 dimGridSum(((header.paddedLength/2)+1 + dimBlockSum.x - 1) / dimBlockSum.x);
        sum_across_channels<<<dimGridSum, dimBlockSum>>>(deviceData_float2_dedispersed, deviceData_float2_single_spectrum, header.nchans, (header.paddedLength/2)+1);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel2, 0);
        cudaEventSynchronize(stopKernel2);
        float elapsedTime2;
        cudaEventElapsedTime(&elapsedTime2, startKernel2, stopKernel2);
        printf("Sum kernel (naive implementation, single DM) time:\t\t%lf s\n", elapsedTime2 / 1000.0);

        // time the kernel
        cudaEvent_t startKernel2_smem, stopKernel2_smem;
        cudaEventCreate(&startKernel2_smem);
        cudaEventCreate(&stopKernel2_smem);
        cudaEventRecord(startKernel2_smem, 0);

        // sum across channels using smem version
        dim3 dimBlockSum_smem(8, 128);
        dim3 dimGridSum_smem(((header.paddedLength/2)+1 + dimBlockSum_smem.x - 1) / dimBlockSum_smem.x, 32);
        sum_across_channels_smem<<<dimGridSum_smem, dimBlockSum_smem, 128 * 8 * sizeof(float2)>>>(deviceData_float2_dedispersed, deviceData_float2_dedispersed_block, (header.paddedLength/2)+1, header.nchans);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel2_smem, 0);
        cudaEventSynchronize(stopKernel2_smem);
        float elapsedTime2_smem;
        cudaEventElapsedTime(&elapsedTime2_smem, startKernel2_smem, stopKernel2_smem);
        printf("Sum kernel (smem implementation, 32 DMs) time:\t\t\t%lf s\n", elapsedTime2_smem / 1000.0);

    //}


    // transfer deviceData_float2_single_spectrum to host and write as csv
    float2* hostData_float2_single_spectrum = (float2*) malloc(((header.paddedLength/2)+1) * sizeof(float2));
    cudaMemcpy(hostData_float2_single_spectrum, deviceData_float2_single_spectrum, ((header.paddedLength/2)+1) * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    FILE *fp;
    fp = fopen("single_spectrum_dedispersed.csv", "w+");
    for (int i = 0; i < (header.paddedLength/2)+1; i++){
        fprintf(fp, "%f, %f\n", hostData_float2_single_spectrum[i].x, hostData_float2_single_spectrum[i].y);
    }





    // free memory
    cudaFree(deviceData_float2_raw);
    cudaFree(deviceData_float2_dedispersed);
    cudaFree(deviceData_float2_single_spectrum);
    free(hostFilterbank.data);
    free(timeShifts);

    // check cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 3: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // stop timing
    gettimeofday(&end, NULL);
    printf("Total time:\t\t\t%lf s\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);
    return 0;
}