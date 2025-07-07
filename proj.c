%%writefile proj.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#pragma pack(push,1)
typedef struct {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

unsigned char* loadBMP(const char* filename, int *width, int *height, int *rowSize) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    BITMAPFILEHEADER fileHeader;
    fread(&fileHeader, sizeof(fileHeader), 1, f);
    if (fileHeader.bfType != 0x4D42) {
        fclose(f);
        return NULL;
    }

    BITMAPINFOHEADER infoHeader;
    fread(&infoHeader, sizeof(infoHeader), 1, f);

    if (infoHeader.biBitCount != 24) {
        fclose(f);
        return NULL;
    }

    *width = infoHeader.biWidth;
    *height = infoHeader.biHeight;
    *rowSize = (*width * 3 + 3) & (~3);

    fseek(f, fileHeader.bfOffBits, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(*rowSize * (*height));
    fread(data, 1, *rowSize * (*height), f);
    fclose(f);
    return data;
}

void writeBMP(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    int rowSize = (width * 3 + 3) & (~3);
    int dataSize = rowSize * height;

    BITMAPFILEHEADER fileHeader = {0};
    BITMAPINFOHEADER infoHeader = {0};

    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = sizeof(fileHeader) + sizeof(infoHeader) + dataSize;
    fileHeader.bfOffBits = sizeof(fileHeader) + sizeof(infoHeader);

    infoHeader.biSize = sizeof(infoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biSizeImage = dataSize;

    fwrite(&fileHeader, sizeof(fileHeader), 1, f);
    fwrite(&infoHeader, sizeof(infoHeader), 1, f);
    fwrite(data, 1, dataSize, f);
    fclose(f);
}

__global__ void sobelKernel(unsigned char* gray, unsigned char* result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx = y * width + x;

        int Gx = -gray[(y-1)*width + (x-1)] - 2 * gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)]
                 + gray[(y-1)*width + (x+1)] + 2 * gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];
        int Gy = -gray[(y-1)*width + (x-1)] - 2 * gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)]
                 + gray[(y+1)*width + (x-1)] + 2 * gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];

        int val = abs(Gx) + abs(Gy);
        if (val > 255) val = 255;
        result[idx] = (unsigned char)val;
    }
}

void rgbToGray(unsigned char* img, unsigned char* gray, int width, int height, int rowSize) {
    for (int y = 0; y < height; y++) {
        unsigned char* row = img + y * rowSize;
        for (int x = 0; x < width; x++) {
            unsigned char b = row[3*x + 0];
            unsigned char g = row[3*x + 1];
            unsigned char r = row[3*x + 2];
            gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

void grayToRGB(unsigned char* gray, unsigned char* img, int width, int height, int rowSize) {
    for (int y = 0; y < height; y++) {
        unsigned char* row = img + y * rowSize;
        for (int x = 0; x < width; x++) {
            unsigned char val = gray[y * width + x];
            row[3*x + 0] = val;
            row[3*x + 1] = val;
            row[3*x + 2] = val;
        }
    }
}

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after %s: %s\n", message, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./sobel_cuda input.bmp output.bmp\n");
        return 1;
    }

    int width, height, rowSize;
    unsigned char* img = loadBMP(argv[1], &width, &height, &rowSize);
    if (!img) {
        printf("Failed to load image or unsupported format\n");
        return 1;
    }

    int size = width * height;
    unsigned char *gray = (unsigned char*)malloc(size);
    unsigned char *result = (unsigned char*)malloc(size);

    rgbToGray(img, gray, width, height, rowSize);

    unsigned char *d_gray, *d_result;
    cudaMalloc((void**)&d_gray, size);
    checkCudaError("cudaMalloc d_gray");
    cudaMalloc((void**)&d_result, size);
    checkCudaError("cudaMalloc d_result");

    cudaMemcpy(d_gray, gray, size, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy to d_gray");

    dim3 block(16, 16);
    dim3 grid((width + 15)/16, (height + 15)/16);
    sobelKernel<<<grid, block>>>(d_gray, d_result, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution");

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy to host");

    grayToRGB(result, img, width, height, rowSize);
    writeBMP(argv[2], img, width, height);

    cudaFree(d_gray);
    cudaFree(d_result);
    free(img);
    free(gray);
    free(result);

    printf("Sobel filter applied and saved to %s\n", argv[2]);
    return 0;
}
