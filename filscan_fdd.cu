#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define ROTATION_SMEM_WIDTH 128
#define ROTATION_SMEM_HEIGHT 16

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

struct SharedMemory2D {
    float2* data;
    __device__ float2* operator[](int idx) {
        return &data[idx * ROTATION_SMEM_WIDTH];
    }
};

__global__ void rotate_spectrum_smem(float2* deviceData_float2, float2* deviceData_output_float2, long nsamps, float FFTbinWidth, long nchans, float DMstart, float DMstep){
    //extern __shared__ float2 input[ROTATION_SMEM_HEIGHT][ROTATION_SMEM_WIDTH];   // ROTATION_SMEM_HEIGHT channels, ROTATION_SMEM_WIDTH samples per channel
    //extern __shared__ float2 output[ROTATION_SMEM_HEIGHT][ROTATION_SMEM_WIDTH];  // ROTATION_SMEM_HEIGHT DMs, ROTATION_SMEM_WIDTH samples per channel

    extern __shared__ float2 sharedMemory[];

    SharedMemory2D input = { &sharedMemory[0] };
    SharedMemory2D output = { &sharedMemory[ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH] };  // Offset by the size of the input array

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
    // begin overall program timer using gettimeofday()
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // begin load data timer
    struct timeval load_start, load_end;
    gettimeofday(&load_start, NULL);

    printf("%s", filscan_frame);

    if (argc != 2) {
        printf("Usage: %s <file_name>\n", argv[0]);
        return 1;
    }


    struct header header;
    readHeader(argv[1], &header);
    printHeaderStruct(&header);

    struct hostFilterbank hostFilterbank;
    hostFilterbank.header = header;
    hostFilterbank.data = (uint8_t*) malloc(header.dataSize * sizeof(uint8_t));
    readFilterbankData(&header, &hostFilterbank);

    // end load data timer using gettimeofday()
    gettimeofday(&load_end, NULL);
    double load_elapsed = (load_end.tv_sec - load_start.tv_sec) + (load_end.tv_usec - load_start.tv_usec) / 1000000.0;



    // initialise all GPU arrays here
    // start cuda malloc timer using gettimeofday()
    struct timeval malloc_start, malloc_end;
    gettimeofday(&malloc_start, NULL);

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
    long dataLength = nchans * nextPowerOf2;

    printf("Data length:\t\t\t%ld bytes\n", dataLength);

    u_int8_t* deviceData_uint8_t;
    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));

    float* deviceData_float;
    cudaMalloc((void**)&deviceData_float, dataLength * sizeof(float));

    // check errors after mallocs
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("device data CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemset(deviceData_float, 0, dataLength * sizeof(float));

    // make a float2 copy of the array for cufft output
    float2* deviceData_float2;
    cudaMalloc((void**)&deviceData_float2, ((nextPowerOf2/2)+1) * nchans * sizeof(float2));

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("float2 CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }





    // print memory utilisation statistics
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("\nGPU memory usage:\t\tused = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);


    float2* deviceData_float2_summed;
    cudaMalloc((void**)&deviceData_float2_summed, ((nextPowerOf2/2)+1) * sizeof(float2));

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("malloc2 CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    // end cuda malloc timer using gettimeofday()
    gettimeofday(&malloc_end, NULL);

    // start data transfer timer using gettimeofday()
    struct timeval transfer_start, transfer_end;
    gettimeofday(&transfer_start, NULL);

    cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("H2D CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // end data transfer timer using gettimeofday()
    gettimeofday(&transfer_end, NULL);

    // start transpose cast timer using gettimeofday()
    struct timeval transpose_start, transpose_end;
    gettimeofday(&transpose_start, NULL);

    // transpose and cast
    // copy into float array where each channel is padded to the next highest power of 2 length and set to 0


    dim3 dimBlock(32, 32);
    dim3 dimGrid((nsamps + dimBlock.x - 1) / dimBlock.x, (nchans + dimBlock.y - 1) / dimBlock.y);
    transpose_and_cast_uint8_t_to_padded_float<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_float, nchans, nsamps, nextPowerOf2);
    cudaDeviceSynchronize();
    cudaFree(deviceData_uint8_t);
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("transpose CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
        
    // end transpose cast timer using gettimeofday()
    gettimeofday(&transpose_end, NULL);


    // start FFT timer using gettimeofday()
    struct timeval fft_start, fft_end;
    gettimeofday(&fft_start, NULL);

    // cufft each channel, storing the output in the float2 array
    cufftHandle plan;
    cufftPlan1d(&plan, nextPowerOf2, CUFFT_R2C, nchans);
    cufftExecR2C(plan, deviceData_float, deviceData_float2);
    cudaDeviceSynchronize();
    cudaFree(deviceData_float);
    cudaDeviceSynchronize();

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("cufft CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    // end FFT timer using gettimeofday()
    gettimeofday(&fft_end, NULL);
    
    // start rotation timer using gettimeofday()
    struct timeval rotate_start, rotate_end;
    gettimeofday(&rotate_start, NULL);

    // compute the time shifts for each channel
    float* timeShifts = (float*) malloc(nchans * sizeof(float));
    compute_time_shifts(timeShifts, header.fch1, header.foff, nchans, 1.0);

    // copy the time shifts to the device constant memory
    cudaMemcpyToSymbol(cachedTimeShiftsPerDM, timeShifts, nchans * sizeof(float));
    
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("symbol CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    float DM = 50;

    // rotate the spectrum channelwise
    
    float DMstep = 0.1;

    dim3 dimBlock2(ROTATION_SMEM_WIDTH, 1);
    dim3 dimGrid2(((nextPowerOf2/2)+1 + dimBlock2.x - 1) / dimBlock2.x, nchans / ROTATION_SMEM_HEIGHT);

    cudaDeviceSynchronize();
    // get last cuda error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("before kernel CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    printf("Launching kernel with arguments:\n");
    printf("dimGrid2: %d, %d\n", dimGrid2.x, dimGrid2.y);
    printf("dimBlock2: %d, %d\n", dimBlock2.x, dimBlock2.y);
    printf("smem: %ld\n", 2 * ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH * sizeof(float2));
    printf("deviceData_float2: %p\n", deviceData_float2);
    printf("deviceData_float2: %p\n", deviceData_float2);
    printf("((nextPowerOf2/2)+1): %ld\n", ((nextPowerOf2/2)+1));
    printf("FFTbinWidth: %f\n", FFTbinWidth);
    printf("nchans: %ld\n", nchans);
    printf("DM: %f\n", DM);
    printf("DMstep: %f\n", DMstep);

    cudaFuncSetAttribute(rotate_spectrum_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 99*1024);
    rotate_spectrum_smem<<<dimGrid2, dimBlock2, 2 * ROTATION_SMEM_HEIGHT * ROTATION_SMEM_WIDTH * sizeof(float2)>>>(deviceData_float2, deviceData_float2, ((nextPowerOf2/2)+1), FFTbinWidth, nchans, DM, DMstep);
    cudaDeviceSynchronize();
    
    // get last cuda error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("after kernel CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // end rotation timer using gettimeofday()
    gettimeofday(&rotate_end, NULL);

    // start free memory timer using gettimeofday()
    struct timeval free_start, free_end;
    gettimeofday(&free_start, NULL);

    // free memory
    cudaFree(deviceData_uint8_t);
    cudaFree(deviceData_float);
    cudaFree(deviceData_float2);
    //cudaFree(deviceTimeShifts);
    free(hostFilterbank.data);
    free(timeShifts);

    // end free memory timer using gettimeofday()
    gettimeofday(&free_end, NULL);

    // end overall program timer using gettimeofday()
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // print timing statistics
    printf("\nLoad data time:\t\t\t%lf s\n", load_elapsed);
    printf("Malloc time:\t\t\t%lf s\n", (malloc_end.tv_sec - malloc_start.tv_sec) + (malloc_end.tv_usec - malloc_start.tv_usec) / 1000000.0);
    printf("Data transfer time H2D:\t\t%lf s\n", (transfer_end.tv_sec - transfer_start.tv_sec) + (transfer_end.tv_usec - transfer_start.tv_usec) / 1000000.0);
    printf("Transpose and cast time:\t%lf s\n", (transpose_end.tv_sec - transpose_start.tv_sec) + (transpose_end.tv_usec - transpose_start.tv_usec) / 1000000.0);
    printf("FFT time:\t\t\t%lf s\n", (fft_end.tv_sec - fft_start.tv_sec) + (fft_end.tv_usec - fft_start.tv_usec) / 1000000.0);
    printf("Rotation time:\t\t\t%lf s\n", (rotate_end.tv_sec - rotate_start.tv_sec) + (rotate_end.tv_usec - rotate_start.tv_usec) / 1000000.0);
    printf("Free memory time:\t\t%lf s\n", (free_end.tv_sec - free_start.tv_sec) + (free_end.tv_usec - free_start.tv_usec) / 1000000.0);
    printf("\nTotal elapsed time:\t\t%lf s\n", elapsed);


    return 0;
}
