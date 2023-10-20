#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cufft.h>

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

//__global__ void rotate_spectrum_smem(){
//}

//__global__ void reassemble_fragments(){
//}

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
    float2* deviceData_float2;

    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));
    cudaMalloc((void**)&deviceData_float, header.nchans * header.paddedLength * sizeof(float));
    cudaMalloc((void**)&deviceData_float2, ((header.paddedLength/2)+1) * header.nchans * sizeof(float2));
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after first mallocs:\t\t%ld MB\n", availableMemory / 1024 / 1024);

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


    // cufft each channel, storing the output in the float2 array
    cufftHandle plan;
    cufftPlan1d(&plan, header.paddedLength, CUFFT_R2C, header.nchans);
    cufftExecR2C(plan, deviceData_float, deviceData_float2);
    cudaDeviceSynchronize();
    cudaFree(deviceData_float);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free float:\t\t%ld MB\n", availableMemory / 1024 / 1024);

    // compute the time shifts for each channel
    float* timeShifts = (float*) malloc(header.nchans * sizeof(float));
    compute_time_shifts(timeShifts, header.fch1, header.foff, header.nchans, 1.0);
    cudaMemcpyToSymbol(cachedTimeShiftsPerDM, timeShifts, header.nchans * sizeof(float));


    






    // free memory
    cudaFree(deviceData_float2);
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
