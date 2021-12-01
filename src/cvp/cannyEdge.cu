#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

#include "cannyEdge.cuh"
#include "helper.cuh"

#define CONV_KERNEL_SIZE 5
__constant__ float CK[5][5];

__global__ void test_print()
{
  printf("Hello World!\n");
}

void wrap_test_print()
{
  test_print<<<1, 1>>>();
  return;
}

__global__ void conv2D(uint8_t* in, uint8_t* out, unsigned int width, unsigned int height, unsigned int pitchIn, unsigned int pitchOut)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if(row < height && col < width)
    out[row * pitchOut + col] = in[row * pitchIn + col];
}

void cvp::cuda::convolution2D(unsigned char* imageCPU, unsigned int pitchCPU, unsigned int imageWidth, unsigned int imageHeight, unsigned int blockSize)
{
  if(blockSize > imageWidth || blockSize > imageHeight)
  {
    LOG_ERROR("BlockSize cannot be bigger than imageSize");
    return;
  }

  size_t pitchInGPU;
  uint8_t* inputImageGPU;
  checkCudaErrors(cudaMallocPitch(&inputImageGPU, &pitchInGPU, imageWidth * 3, imageHeight));

  size_t pitchOutGPU;
  uint8_t* outputImageGPU;
  checkCudaErrors(cudaMallocPitch(&outputImageGPU, &pitchOutGPU, imageWidth * 3, imageHeight));

  // copy to device
  checkCudaErrors(cudaMemcpy2D(inputImageGPU, pitchInGPU, imageCPU, pitchCPU, imageWidth * 3, imageHeight, cudaMemcpyHostToDevice));

  float timeMs;

  dim3 grid(imageWidth/blockSize, imageHeight/blockSize, 1);
  dim3 blocks(blockSize, blockSize, 1);
  conv2D<<<grid, blocks>>>(inputImageGPU, outputImageGPU, imageWidth, imageHeight, pitchInGPU, pitchOutGPU);

  // copy back to host
  cudaMemcpy2D(imageCPU, imageWidth * 3, outputImageGPU, pitchOutGPU, imageWidth * 3, imageHeight, cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaFree(inputImageGPU));
  checkCudaErrors(cudaFree(outputImageGPU));

  return;
}

#endif

/*
 // checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&outputImage), memImageSize));

Mat mat_BGR(height, width, CV_8UC3);
Mat mat_BGR_disp(height, width, CV_8UC3);

mat_BGR = RGBCamera.CaptureMat(1); // wrapper for the RGB camera that grabs the image

// copy to device
cudaMemcpy2D(image_BGR_gpu, pitch_BGR, mat_BGR.ptr(), width, matrixLenBGR, height, cudaMemcpyHostToDevice);


namedWindow("Display window", WINDOW_AUTOSIZE);
imshow("Display window", mat_BGR_disp);*/