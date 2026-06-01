# GPU-Accelerated-Sobel-Edge-Detection-for-Image-Processing-Using-CUDA

# DEVELOPED BY :
NAME : MERIL GOLDLINA A

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
