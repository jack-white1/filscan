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

void compute_time_shifts(double* timeShifts, double f1, double foff, int nchans, double DM, double FFTbinWidth) {
    for (int i = 0; i < nchans; i++) {
        double f2 = f1 + foff * i;

        // convert to GHz
        double f1_GHz = f1 / 1000.0;
        double f2_GHz = f2 / 1000.0;
        double k = 4.148808;

        // compute the time shift in ms
        double timeShift_ms = k * DM * (1.0 / (f1_GHz * f1_GHz) - 1.0 / (f2_GHz * f2_GHz));

        // convert to seconds
        timeShifts[i] = - timeShift_ms / 1000.0;
    }
}



__global__ void transpose_and_pad_uint8_t(uint8_t* input, uint8_t* output, int nchans, int input_nsamps, int output_nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < input_nsamps && y < nchans) {
        output[y * output_nsamps + x] = input[x * nchans + y];
        //output[y * output_nsamps + x] = 40;
    }
}

static __constant__ int channelCopies[8192];
static __constant__ int cumulativeChannelCopies[8192];

__global__ void stretch_uint8_t_channels(uint8_t* input, uint8_t* output, int nchans, int nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int numCopies = channelCopies[y];
    int outputRowIndex = cumulativeChannelCopies[y] - channelCopies[y];

    if (x < nsamps && y < nchans) {
        for (int i = 0; i < numCopies; i++) {
            output[outputRowIndex * nsamps + x] = input[y * nsamps + x];
            outputRowIndex++;
        }
    }
}

__global__ void copy_uint8_t_array_to_float(uint8_t* input, float* output, long nchans, long nsamps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nsamps && y < nchans) {
        output[y * nsamps + x] = (float) input[y * nsamps + x];
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
    u_int8_t* deviceData_padded_uint8_t;

    size_t availableMemory, totalMemory;

    cudaMalloc((void**)&deviceData_uint8_t, header.dataSize * sizeof(uint8_t));
    cudaMalloc((void**)&deviceData_padded_uint8_t, header.nchans * header.paddedLength * sizeof(uint8_t));
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("\nAvailable memory after first mallocs:\t\t%ld MB\n", availableMemory / 1024 / 1024);



    // set padded to 0
    cudaMemset(deviceData_padded_uint8_t, 0, header.nchans * header.paddedLength * sizeof(uint8_t));

    cudaMemcpy(deviceData_uint8_t, hostFilterbank.data, header.dataSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    // transpose and pad
    dim3 dimBlock(32, 32);
    dim3 dimGrid((header.nsamp + dimBlock.x - 1) / dimBlock.x, (header.nchans + dimBlock.y - 1) / dimBlock.y);
    transpose_and_pad_uint8_t<<<dimGrid, dimBlock>>>(deviceData_uint8_t, deviceData_padded_uint8_t, header.nchans, header.nsamp, header.paddedLength);

    printf("transpose_and_pad_uint8_t kernel called with the following arguments:\n");
    printf("dimGrid.x:\t\t\t%d\n", dimGrid.x);
    printf("dimGrid.y:\t\t\t%d\n", dimGrid.y);
    printf("dimBlock.x:\t\t\t%d\n", dimBlock.x);
    printf("dimBlock.y:\t\t\t%d\n", dimBlock.y);
    printf("header.nchans:\t\t\t%d\n", header.nchans);
    printf("header.nsamp:\t\t\t%ld\n", header.nsamp);
    printf("header.paddedLength:\t\t%ld\n", header.paddedLength);
    
    cudaDeviceSynchronize();
    cudaMemGetInfo(&availableMemory, &totalMemory);
    printf("Available memory after free uint8:\t\t%ld MB\n", availableMemory / 1024 / 1024);

    // compute the time shifts for each channel
    double* timeShifts = (double*) malloc(header.nchans * sizeof(double));
    compute_time_shifts(timeShifts, (double)header.fch1, (double)header.foff, header.nchans, (double)1.0, (double) (1.0 / (header.tsamp * header.paddedLength)));

    int* numCopiesArray = (int*) malloc(header.nchans * sizeof(int));
    // set numCopiesArray to all ones
    for (int i = 0; i < header.nchans; i++) {
        numCopiesArray[i] = 1;
    }

    double* timeShiftDifferences = (double*) malloc((header.nchans-1) * sizeof(double));
    //for (int i = 0; i < header.nchans-1; i++) {
    double gradientRatioSum = 0.0f;
    long totalExtraChannels = 0;
    for (int i = 0; i < header.nchans-1; i++) {
        timeShiftDifferences[i] = timeShifts[i+1] - timeShifts[i];
        gradientRatioSum += (timeShiftDifferences[i] / timeShiftDifferences[0])-1;
        while (gradientRatioSum > 1.0f) {
            gradientRatioSum -= 1.0f;
            totalExtraChannels++;
            numCopiesArray[i]++;
        }
    }

    int cumulativeNumCopies = 0;
    int* cumulativeNumCopiesArray = (int*) malloc(header.nchans * sizeof(int));
    for (int i = 0; i < header.nchans; i++) {
        cumulativeNumCopies += numCopiesArray[i];
        cumulativeNumCopiesArray[i] = cumulativeNumCopies;
    }



    printf("Total extra channels:\t\t%ld\n", totalExtraChannels);

    long numChannelsStretched = (long)header.nchans + (long)totalExtraChannels;
    long numChannelsStretchedPadded = numChannelsStretched;

    // increase numChannelsStretchedPadded to the next power of 2
    long nextPowerOf2 = 1;
    while (nextPowerOf2 < numChannelsStretchedPadded) {
        nextPowerOf2 *= 2;
    }
    numChannelsStretchedPadded = nextPowerOf2;

    printf("Number of channels stretched:\t%ld\n", numChannelsStretched);
    printf("Number of channels stretched and padded to next power of 2:\t%ld\n", numChannelsStretchedPadded);

    // allocate memory for the stretched data
    uint8_t* stretchedData;
    cudaMalloc((void**)&stretchedData, numChannelsStretchedPadded * header.paddedLength * sizeof(uint8_t));
    printf("Size of stretched data:\t\t%ld bytes\n", numChannelsStretchedPadded * header.paddedLength * sizeof(uint8_t));


    // stretch the channels
    cudaMemcpyToSymbol(channelCopies, numCopiesArray, header.nchans * sizeof(int));
    cudaMemcpyToSymbol(cumulativeChannelCopies, cumulativeNumCopiesArray, header.nchans * sizeof(int));

    // get last cuda error, print custom error message for this part of the program
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error symbol: %s\n", cudaGetErrorString(error));
        return 1;
    }

    dim3 dimBlock2(1024, 1);
    dim3 dimGrid2((header.paddedLength + dimBlock2.x - 1) / dimBlock2.x, header.nchans);
    stretch_uint8_t_channels<<<dimGrid2, dimBlock2>>>(deviceData_padded_uint8_t, stretchedData, header.nchans, header.paddedLength);

    printf("stretch_uint8_t_channels kernel called with the following arguments:\n");
    printf("dimGrid.x:\t\t\t%d\n", dimGrid2.x);
    printf("dimGrid.y:\t\t\t%d\n", dimGrid2.y);
    printf("dimBlock.x:\t\t\t%d\n", dimBlock2.x);
    printf("dimBlock.y:\t\t\t%d\n", dimBlock2.y);
    printf("header.nchans:\t\t\t%d\n", header.nchans);
    printf("header.paddedLength:\t\t%ld\n", header.paddedLength);

    cudaDeviceSynchronize();

    // get last cuda error, print custom error message for this part of the program
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error stretch kernel: %s\n", cudaGetErrorString(error));
        return 1;
    }
    

    // copy stretched data back to host
    uint8_t* stretchedDataHost = (uint8_t*) malloc(numChannelsStretchedPadded * header.paddedLength * sizeof(uint8_t));
    cudaMemcpy(stretchedDataHost, stretchedData, numChannelsStretchedPadded * header.paddedLength * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // get last cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error memcpy: %s\n", cudaGetErrorString(error));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();


    // copy stretched data to float array
    float* stretchedDataFloat;
    cudaMalloc((void**)&stretchedDataFloat, numChannelsStretchedPadded * header.paddedLength * sizeof(float));
    dim3 dimBlock3(1024, 1);
    dim3 dimGrid3((header.paddedLength + dimBlock3.x - 1) / dimBlock3.x, numChannelsStretchedPadded);
    printf("Calling copy_uint8_t_array_to_float kernel with the following arguments:\n");
    printf("dimGrid.x:\t\t\t%d\n", dimGrid3.x);
    printf("dimGrid.y:\t\t\t%d\n", dimGrid3.y);
    printf("dimBlock.x:\t\t\t%d\n", dimBlock3.x);
    printf("dimBlock.y:\t\t\t%d\n", dimBlock3.y);
    printf("numChannelsStretchedPadded:\t%ld\n", numChannelsStretchedPadded);
    printf("header.paddedLength:\t\t%ld\n", header.paddedLength);
    printf("Maximum index that will be accessed:\t%ld\n", numChannelsStretchedPadded * header.paddedLength);

    copy_uint8_t_array_to_float<<<dimGrid3, dimBlock3>>>(stretchedData, stretchedDataFloat, numChannelsStretchedPadded, header.paddedLength);
    cudaDeviceSynchronize();

    // get last cuda error
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error copy_uint8_t_array_to_float: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // make output array for 2D FFT of stretchedDataFloat
    cufftComplex* stretchedDataFFT;
    cudaMalloc((void**)&stretchedDataFFT, numChannelsStretchedPadded * (1+header.paddedLength/2) * sizeof(cufftComplex));

    // create FFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, numChannelsStretchedPadded, header.paddedLength, CUFFT_R2C);

    // execute FFT
    cufftExecR2C(plan, stretchedDataFloat, stretchedDataFFT);

    // check cuda errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error cufftExecR2C: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // write cumulativeNumCopiesArray and numCopies array to csv
    FILE *csvFile3 = fopen("numCopies.csv", "w");
    fprintf(csvFile3, "channel,numCopies,cumulativeNumCopies\n");
    for (int i = 0; i < header.nchans; i++) {
        fprintf(csvFile3, "%d,%d,%d\n", i, numCopiesArray[i], cumulativeNumCopiesArray[i]);
    }

    // write stretched data to csv file in format row, column, value
    FILE *csvFile = fopen("stretchedData.csv", "w");
    for (int i = 0; i < numChannelsStretchedPadded; i++) {
        for (int j = 0; j < header.paddedLength; j++) {
            fprintf(csvFile, "%d,%d,%d\n", i, j, stretchedDataHost[i * header.paddedLength + j]);
        }
    }

    // write to csv
    FILE *csvFile1 = fopen("data.csv", "w");
    for (int i = 0; i < header.nchans; i++) {
        for (int j = 0; j < header.nsamp; j++) {
            fprintf(csvFile1, "%d,%d,%d\n", i, j, hostFilterbank.data[i * header.nsamp + j]);
        }
    }

    // copy stretchedDataFFT back to host
    cufftComplex* stretchedDataFFTHost = (cufftComplex*) malloc(numChannelsStretchedPadded * (1+header.paddedLength/2) * sizeof(cufftComplex));
    cudaMemcpy(stretchedDataFFTHost, stretchedDataFFT, numChannelsStretchedPadded * (1+header.paddedLength/2) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // write stretchedDataFFT to csv file in format row, column, complex magnitude
    FILE *csvFile4 = fopen("stretchedDataFFT.csv", "w");
    for (int i = 0; i < numChannelsStretchedPadded; i++) {
        for (int j = 0; j < (1+header.paddedLength/2); j++) {
            fprintf(csvFile4, "%d,%d,%lf\n", i, j, sqrt(stretchedDataFFTHost[i * (1+header.paddedLength/2) + j].x * stretchedDataFFTHost[i * (1+header.paddedLength/2) + j].x + stretchedDataFFTHost[i * (1+header.paddedLength/2) + j].y * stretchedDataFFTHost[i * (1+header.paddedLength/2) + j].y));
        }
    }



    // free host memory
    free(hostFilterbank.data);
    free(timeShifts);
    free(numCopiesArray);
    free(timeShiftDifferences);
    free(cumulativeNumCopiesArray);
    free(stretchedDataHost);
    free(stretchedDataFFTHost);

    // free device memory
    cudaFree(deviceData_uint8_t);
    cudaFree(deviceData_padded_uint8_t);
    cudaFree(stretchedData);
    cudaFree(stretchedDataFloat);
    cudaFree(stretchedDataFFT);

    // stop timing
    gettimeofday(&end, NULL);
    printf("Total time:\t\t\t%lf s\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);
    return 0;
}
