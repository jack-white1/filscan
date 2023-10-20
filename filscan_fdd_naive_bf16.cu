#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <cuda_bf16.h>
#include <cufftXt.h>

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

__global__ void transpose_and_cast_uint8_t_to_padded_bfloat16(uint8_t* deviceData_uint8_t, __nv_bfloat16* deviceData_bfloat16, int nchans, int input_nsamps, int output_nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < input_nsamps && y < nchans) {
        deviceData_bfloat16[y * output_nsamps + x] = 	__ushort2bfloat16_rz(deviceData_uint8_t[x * nchans + y]);
    }
}

static __constant__ float cachedTimeShiftsPerDM[4096];

__global__ void rotate_spectrum(__nv_bfloat162* inputArray, __nv_bfloat162* outputArray, long nchans, long nsamps, float DM){
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    long y = blockIdx.y;

    long outputIndex = y * nsamps + x;
    //printf("outputIndex: %ld\n", outputIndex);
    //outputIndex = 0;


    if (x < nsamps-1 && y < nchans-1) {
        float phase = x * DM * cachedTimeShiftsPerDM[y];
        __nv_bfloat162 input = inputArray[outputIndex];
        __nv_bfloat162 output;
        float s, c;
        sincosf(phase, &s, &c);
        output.x = __float2bfloat16(__bfloat162float(input.x) * c - __bfloat162float(input.y) * s);
        output.y = __float2bfloat16(__bfloat162float(input.x) * s + __bfloat162float(input.y) * c);
        outputArray[outputIndex] = output;
    }
}


__global__ void sum_across_channels(__nv_bfloat162* inputArray, __nv_bfloat162* outputArray, long nchans, long nsamps){
    long x = blockIdx.x * blockDim.x + threadIdx.x;
    //x = 0;

    __nv_bfloat162 sum;
    sum.x = 0.0;
    sum.y = 0.0;

    if (x < nsamps) {
        for (long y = 0; y < nchans; y++) {
            __nv_bfloat162 value = inputArray[y * nsamps + x];
            sum.x += value.x;
            sum.y += value.y;
        }
        outputArray[x] = sum;
    }
    
    
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
    __nv_bfloat16* deviceData_bfloat16;
    __nv_bfloat162* deviceData___nv_bfloat162_raw;
    __nv_bfloat162* deviceData___nv_bfloat162_dedispersed;
    __nv_bfloat162* deviceData___nv_bfloat162_single_spectrum;

    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));
    cudaMalloc((void**)&deviceData_bfloat16, header.nchans * header.paddedLength * sizeof(__nv_bfloat16));
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("\nAvailable memory after first mallocs:\t\t%ld MB\n", availableMemory / 1024 / 1024);



    cudaMemset(deviceData_bfloat16, 0, header.nchans * header.paddedLength * sizeof(__nv_bfloat16));

    cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    // transpose and cast
    dim3 dimBlock(32, 32);
    dim3 dimGrid((header.paddedLength + dimBlock.x - 1) / dimBlock.x, (header.nchans + dimBlock.y - 1) / dimBlock.y);
    transpose_and_cast_uint8_t_to_padded_bfloat16<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_bfloat16, header.nchans, header.nsamp, header.paddedLength);
    cudaDeviceSynchronize();
    cudaFree(deviceData_uint8_t);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free uint8:\t\t%ld MB\n", availableMemory / 1024 / 1024);


    cudaMalloc((void**)&deviceData___nv_bfloat162_raw, ((header.paddedLength/2)+1) * header.nchans * sizeof(__nv_bfloat162));


    // cufft each channel, storing the output in the __nv_bfloat162 array
    
    // FP32 version
    //cufftHandle plan;
    //cufftPlan1d(&plan, header.paddedLength, CUFFT_R2C, header.nchans);
    //cufftExecR2C(plan, deviceData_bfloat16, deviceData___nv_bfloat162_raw);

    // bfloat16 version
    cufftHandle plan;
    cufftCreate(&plan);
    cudaDataType inputtype = CUDA_R_16BF;
    cudaDataType outputtype = CUDA_C_16BF;
    cudaDataType executiontype = CUDA_C_16BF;

    // Defining the variables based on the assumptions
    int rank = 1;
    long long int n[1] = {header.paddedLength};
    long long int *inembed = NULL;  // assuming default strides
    long long int istride = 1;
    long long int idist = header.paddedLength;
    long long int *onembed = NULL;  // assuming default strides
    long long int ostride = 1;
    long long int odist = header.paddedLength / 2 + 1;
    long long int batch = header.nchans;
    size_t workSize[1];  // Ensure this has enough space if using multiple GPUs

    cufftResult result = cufftXtMakePlanMany(
        plan, rank, n, 
        inembed, istride, idist, 
        inputtype,
        onembed, ostride, odist, 
        outputtype, 
        batch, 
        workSize, 
        executiontype
    );

    // Check result for errors
    if (result != CUFFT_SUCCESS) {
        // Handle error
        printf("CUFFT error: %d\n", result);
        //return;
    }

    result = cufftXtExec(plan, deviceData_bfloat16, deviceData___nv_bfloat162_raw, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // Check result for errors
    if (result != CUFFT_SUCCESS) {
        // Handle error
        printf("CUFFT error: %d\n", result);
        //return;
    }
    

    cudaFree(deviceData_bfloat16);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free float:\t\t%ld MB\n", availableMemory / 1024 / 1024);
    cudaMalloc((void**)&deviceData___nv_bfloat162_dedispersed, ((header.paddedLength/2)+1) * header.nchans * sizeof(__nv_bfloat162));
    cudaMalloc((void**)&deviceData___nv_bfloat162_single_spectrum, ((header.paddedLength/2)+1) * sizeof(__nv_bfloat162));




    // compute the time shifts for each channel
    float* timeShifts = (float*) malloc(header.nchans * sizeof(float));
    compute_time_shifts(timeShifts, header.fch1, header.foff, header.nchans, 1.0, FFTbinWidth);
    cudaMemcpyToSymbol(cachedTimeShiftsPerDM, timeShifts, header.nchans * sizeof(float));

    // get last cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error 1: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float DM = 0;
    float DM_step = 1;

    for (int DM_idx = 0; DM_idx < 1024; DM_idx++){
        DM += DM_step;

        // time the kernel
        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventRecord(startKernel, 0);

        // rotate the spectrum
        dim3 dimBlockRotation(1024, 1);
        dim3 dimGridRotation((header.paddedLength + dimBlockRotation.x - 1) / dimBlockRotation.x, header.nchans);
        rotate_spectrum<<<dimGridRotation, dimBlockRotation>>>(deviceData___nv_bfloat162_raw, deviceData___nv_bfloat162_dedispersed, (long)header.nchans, (header.paddedLength/2) + 1, DM);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel, 0);
        cudaEventSynchronize(stopKernel);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startKernel, stopKernel);
        printf("\nRotation kernel time:\t\t\t%lf s\n", elapsedTime / 1000.0);


        // time the kernel
        cudaEvent_t startKernel2, stopKernel2;
        cudaEventCreate(&startKernel2);
        cudaEventCreate(&stopKernel2);
        cudaEventRecord(startKernel2, 0);

        // sum across channels
        dim3 dimBlockSum(1024, 1);
        dim3 dimGridSum((header.paddedLength + dimBlockSum.x - 1) / dimBlockSum.x);
        sum_across_channels<<<dimGridSum, dimBlockSum>>>(deviceData___nv_bfloat162_dedispersed, deviceData___nv_bfloat162_single_spectrum, header.nchans, (header.paddedLength/2)+1);
        cudaDeviceSynchronize();

        // stop timing
        cudaEventRecord(stopKernel2, 0);
        cudaEventSynchronize(stopKernel2);
        float elapsedTime2;
        cudaEventElapsedTime(&elapsedTime2, startKernel2, stopKernel2);
        printf("Sum kernel time:\t\t\t%lf s\n", elapsedTime2 / 1000.0);
    }

    // free memory
    cudaFree(deviceData___nv_bfloat162_raw);
    cudaFree(deviceData___nv_bfloat162_dedispersed);
    cudaFree(deviceData___nv_bfloat162_single_spectrum);
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
