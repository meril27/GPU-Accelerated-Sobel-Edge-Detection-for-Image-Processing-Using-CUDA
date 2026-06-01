# GPU-Accelerated-Sobel-Edge-Detection-for-Image-Processing-Using-CUDA

## DEVELOPED BY : MERIL GOLDLINA A

# PROJECT OVERVIEW :
This project implements GPU-accelerated image processing using CUDA for Sobel edge detection. 
The system processes multiple images in batch mode and applies edge detection using parallel computation on the GPU, significantly improving performance compared to CPU-based processing.
The workflow includes loading multiple images, transferring them to GPU memory, applying the Sobel operator using a CUDA kernel, and saving the processed output images. 
The project demonstrates how CUDA can efficiently handle large-scale image processing tasks.


# EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC Google Colab with NVCC Compiler CUDA Toolkit and OpenCV installed. A sample image for testing.

# PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter. Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024). Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV. Discuss the differences in execution time and output quality.

# PROGRAM:

### Step 1: Open Google Colab
Open Google Colab and create a new notebook.

### Step 2: Enable GPU Support
1. Click Runtime
2. Select Change Runtime Type
3. Choose GPU as Hardware Accelerator
4. Click Save

### Step 4 : Install OpenCV
```c
!pip install opencv-python-headless
```
### Step 4 : Upload Images
```c
from google.colab import files
uploaded = files.upload()
```
### Step 3: Create the CUDA file
```c
%%writefile sobelEdgeDetectionFilter.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {

        int idx = y * width + x;

        int Gx =
            -srcImage[(y-1)*width + (x-1)] - 2*srcImage[y*width + (x-1)] - srcImage[(y+1)*width + (x-1)]
            + srcImage[(y-1)*width + (x+1)] + 2*srcImage[y*width + (x+1)] + srcImage[(y+1)*width + (x+1)];

        int Gy =
            -srcImage[(y-1)*width + (x-1)] - 2*srcImage[(y-1)*width + x] - srcImage[(y-1)*width + (x+1)]
            + srcImage[(y+1)*width + (x-1)] + 2*srcImage[(y+1)*width + x] + srcImage[(y+1)*width + (x+1)];

        float mag = sqrtf((float)(Gx * Gx + Gy * Gy));

        if (mag > 255.0f) mag = 255.0f;

        dstImage[idx] = (unsigned char)mag;
    }

    // IMPORTANT: handle borders (safe output)
    if (x < width && y < height) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            dstImage[y * width + x] = 0;
        }
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image
    Mat image = imread("/content/download.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}
```
```c
import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("output_0.jpg", cv2.IMREAD_GRAYSCALE)
cv2_imshow(img)
```
# OUTPUT:
### ORIGINAL :


<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/71656648-40c3-49fb-9bfc-c3a0cc21db58" />


### SOBEL EDGE DETECTION (CUDA) :

<img width="632" height="537" alt="image" src="https://github.com/user-attachments/assets/68da99b4-d547-459b-a55a-3433416c1944" />



# RESULT:
Thus the program has been executed successfully by using CUDA to enhance the performance of Sobel edge detection in image processing tasks through parallel execution on the GPU.
