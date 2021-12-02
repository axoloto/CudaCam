#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

#include "cannyEdge.cuh"
#include "helper.cuh"

#define CONV_KERNEL_SIZE 5
__constant__ float CK[5][5];

__global__ void conv2D(uint8_t *in, uint8_t *out, unsigned int width, unsigned int height, unsigned int pitchIn, unsigned int pitchOut)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < height && col < width)
    out[row * pitchOut + col] = in[row * pitchIn + col];
}

using namespace cvp::cuda;

Convolution2D_RGB::Convolution2D_RGB() : m_isAlloc(false), m_pitchInGPU(0), m_pitchOutGPU(0){};

Convolution2D_RGB::~Convolution2D_RGB()
{
  endAlloc();
};

void Convolution2D_RGB::initAlloc(unsigned int imageWidth, unsigned int imageHeight)
{
  int nbChannels = 3;

  checkCudaErrors(cudaMallocPitch(&m_inImageGPU, &m_pitchInGPU, imageWidth * nbChannels, imageHeight));
  checkCudaErrors(cudaMallocPitch(&m_outImageGPU, &m_pitchOutGPU, imageWidth * nbChannels, imageHeight));

  m_isAlloc = true;
}

void Convolution2D_RGB::run(unsigned char *imageCPU, unsigned int pitchCPU, unsigned int imageWidth, unsigned int imageHeight, unsigned int blockSize)
{
  if (!m_isAlloc)
    initAlloc(imageWidth, imageHeight);

  if (blockSize > imageWidth || blockSize > imageHeight)
  {
    LOG_ERROR("BlockSize cannot be bigger than imageSize");
    return;
  }

  int nbChannels = 3;

  checkCudaErrors(cudaMemcpy2D(m_inImageGPU, m_pitchInGPU, imageCPU, pitchCPU, imageWidth * nbChannels, imageHeight, cudaMemcpyHostToDevice));

  dim3 blocks(blockSize, blockSize, 1);
  dim3 grid((imageWidth * nbChannels) / blockSize, imageHeight / blockSize, 1);
  conv2D<<<grid, blocks>>>(m_inImageGPU, m_outImageGPU, imageWidth * nbChannels, imageHeight, m_pitchInGPU, m_pitchOutGPU);

  cudaMemcpy2D(imageCPU, imageWidth * nbChannels, m_outImageGPU, m_pitchOutGPU, imageWidth * nbChannels, imageHeight, cudaMemcpyDeviceToHost);
}

void Convolution2D_RGB::endAlloc()
{
  checkCudaErrors(cudaFree(m_inImageGPU));
  checkCudaErrors(cudaFree(m_outImageGPU));

  m_isAlloc = false;
}

#endif