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
    printf("fch1:\t\t\t\t%lf MHz\n", header->fch1);
    printf("foff:\t\t\t\t%lf MHz\n", header->foff);
    printf("tsamp:\t\t\t\t%lf s\n", header->tsamp);
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

__global__ void transpose_uint8_t_to_uint16_t(uint8_t* deviceData_in, uint16_t* deviceData_out, int nchans, int nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nsamps && y < nchans) {
        deviceData_out[y * nsamps + x] = deviceData_in[x * nchans + y];
    }
}

/*__global__ void dedispersion_kernel(uint16_t* deviceData, uint32_t* deviceData_dedispersed, int nsamps, int nchans, float tsamp, int maxDM){
    __shared__ uint16_t* sharedData_raw;
    __shared__ uint32_t* sharedData_dedispersed;


    int channelIndex = 0;
    float binsPerDM = cachedBinShifts[channelIndex];


    for (int DM = 0; DM < maxDM; DM++){
        for (int y = 0; y < 4; y++){
            sharedData_dedispersed[threadIdx.x] += sharedData_raw[threadIdx.x + y * blockDim.x];
        }
        // copy out to global memory with atomic add
        atomicAdd(&deviceData_dedispersed[threadIdx.x + DM * nsamps], sharedData_dedispersed[threadIdx.x]);
    }
}*/

static __constant__ float cachedBinShifts[8192];  

void compute_bin_shifts(float* binShifts, float f1, float foff, float tsamp, int nchans, float DM){
    for (int i = 0; i < nchans; i++){
        double f2 = (double) f1 + (double) foff * (double) i;

        // convert to GHz
        double f1_GHz = f1 / 1000.0;
        double f2_GHz = f2 / 1000.0;
        double k = 4.148808;

        // compute the time shift in ms
        double timeShift_ms = k * DM * (1.0 / (f1_GHz * f1_GHz) - 1.0 / (f2_GHz * f2_GHz));

        // convert to seconds
        double timeshift_s = - timeShift_ms / 1000.0;

        // convert to bins
        double binShift = timeshift_s / tsamp;
        binShifts[i] = (float) binShift;
        //printf("binShifts[%d] = %f\n", i, binShifts[i]);
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

        // calculate the time shift per DM for each channel
        float* binShiftsPerDM = (float*) malloc(header.nchans * sizeof(float));
        compute_bin_shifts(binShiftsPerDM, header.fch1, header.foff, header.tsamp, 4096, 1.0);

        // copy to __const__ array
        cudaMemcpyToSymbol(cachedBinShifts, binShiftsPerDM, header.nchans * sizeof(float));

    // end load data timer using gettimeofday()
    gettimeofday(&load_end, NULL);
    double load_elapsed = (load_end.tv_sec - load_start.tv_sec) + (load_end.tv_usec - load_start.tv_usec) / 1000000.0;

    int nDMs = 128;

    // initialise all GPU arrays here
    // start cuda malloc timer using gettimeofday()
    struct timeval malloc_start, malloc_end;
    gettimeofday(&malloc_start, NULL);


        printf("FFT bin width\t\t\t%lf Hz\n", 1.0 / (header.tsamp * header.nsamp));

        float FFTbinWidth = 1.0 / (header.tsamp * header.nsamp);


        long dataLength = header.nchans * header.nsamp;

        printf("Data length:\t\t\t%ld\n", dataLength);

        u_int8_t* deviceData_uint8_t;
        cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));

        // get last cuda error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error deviceData_uint8_t: %s\n", cudaGetErrorString(error));
        }

        u_int16_t* deviceData_uint16_t;
        cudaMalloc((void**)&deviceData_uint16_t, dataLength * sizeof(uint16_t));
        cudaMemset(deviceData_uint16_t, 0, dataLength * sizeof(uint16_t));

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error deviceData_uint16_t: %s\n", cudaGetErrorString(error));
        }

        u_int32_t* deviceData_dedispersed_uint32t;
        cudaMalloc((void**)&deviceData_dedispersed_uint32t, header.nsamp * nDMs * sizeof(uint32_t));

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error deviceData_dedispersed_uint32t: %s\n", cudaGetErrorString(error));
        }

        float* deviceTimeShiftsSeconds;
        cudaMalloc((void**)&deviceTimeShiftsSeconds, header.nchans * sizeof(float));

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error deviceTimeShiftsSeconds: %s\n", cudaGetErrorString(error));
        }



        // print memory utilisation statistics
        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;
        printf("\nGPU memory usage:\t\tused = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

    // end cuda malloc timer using gettimeofday()
    gettimeofday(&malloc_end, NULL);


    // start data transfer timer using gettimeofday()
    struct timeval transfer_start, transfer_end;
    gettimeofday(&transfer_start, NULL);

        cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error deviceData_uint8_t memcpy: %s\n", cudaGetErrorString(error));
        }



    // end data transfer timer using gettimeofday()
    gettimeofday(&transfer_end, NULL);

    // start transpose cast timer using gettimeofday()
    struct timeval transpose_start, transpose_end;
    gettimeofday(&transpose_start, NULL);

        // transpose and cast
        // copy into float array where each channel is padded to the next highest power of 2 length and set to 0

        dim3 dimBlock(32, 32);
        dim3 dimGrid((header.nsamp + dimBlock.x - 1) / dimBlock.x, (header.nchans + dimBlock.y - 1) / dimBlock.y);
        transpose_uint8_t_to_uint16_t<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_uint16_t, header.nchans, header.nsamp);
        cudaDeviceSynchronize();

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error transpose: %s\n", cudaGetErrorString(error));
        }

    // end transpose cast timer using gettimeofday()
    gettimeofday(&transpose_end, NULL);





















    
    // start free memory timer using gettimeofday()
    struct timeval free_start, free_end;
    gettimeofday(&free_start, NULL);

        // free memory
        cudaFree(deviceData_uint8_t);
        cudaFree(deviceData_uint16_t);
        //cudaFree(deviceData_dedispersed_uint32t);
        free(hostFilterbank.data);

        error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error cudaFree: %s\n", cudaGetErrorString(error));
        }


    // end free memory timer using gettimeofday()
    gettimeofday(&free_end, NULL);

    // end overall program timer using gettimeofday()
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // print timing statistics
    printf("\nLoad data time:\t\t\t%lf s\n", load_elapsed);
    printf("Malloc time:\t\t\t%lf s\n", (malloc_end.tv_sec - malloc_start.tv_sec) + (malloc_end.tv_usec - malloc_start.tv_usec) / 1000000.0);
    printf("Data transfer time:\t\t%lf s\n", (transfer_end.tv_sec - transfer_start.tv_sec) + (transfer_end.tv_usec - transfer_start.tv_usec) / 1000000.0);
    printf("Transpose and cast time:\t%lf s\n", (transpose_end.tv_sec - transpose_start.tv_sec) + (transpose_end.tv_usec - transpose_start.tv_usec) / 1000000.0);
    printf("Free memory time:\t\t%lf s\n", (free_end.tv_sec - free_start.tv_sec) + (free_end.tv_usec - free_start.tv_usec) / 1000000.0);
    printf("\nTotal elapsed time:\t\t%lf s\n", elapsed);

    error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error final: %s\n", cudaGetErrorString(error));
    }


    return 0;
}