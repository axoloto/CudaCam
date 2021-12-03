/*
#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

#include "cannyEdge.cuh"
#include "helper.cuh"


using namespace cvp::cuda;
/*
CannyEdge<T>::CannyEdge(unsigned int imageWidth, unsigned int imageHeight) : m_isAlloc(false), m_nbChannels(nbChannels),
                                                                                                   m_imageWidth(imageWidth), m_imageHeight(imageHeight),
                                                                                                   m_pitchInGPU(0), m_pitchOutGPU(0)
{
  initAlloc();
};

CannyEdge<T>::~CannyEdge()
{
  endAlloc();
};

void CannyEdge<T>::initAlloc()
{
  LOG_DEBUG("Start allocating image memory in GPU");

  checkCudaErrors(cudaMallocPitch(&m_inImageGPU, &m_pitchInGPU, m_imageWidth * m_nbChannels, m_imageHeight));
  checkCudaErrors(cudaMallocPitch(&m_outImageGPU, &m_pitchOutGPU, m_imageWidth * m_nbChannels, m_imageHeight));

  LOG_DEBUG("End allocating image memory in GPU");

  m_isAlloc = true;
}

void CannyEdge<T>::endAlloc()
{
  LOG_DEBUG("Start deallocating image memory in GPU");

  checkCudaErrors(cudaFree(m_inImageGPU));
  checkCudaErrors(cudaFree(m_outImageGPU));

  LOG_DEBUG("End deallocating image memory in GPU");

  m_isAlloc = false;
}

void CannyEdge<T>::loadImage(unsigned char *inImageCPU, size_t inImageCPUPitch)
{
  if (!inImageCPU)
  {
    LOG_ERROR("Cannot load image to GPU");
    return;
  }

  LOG_DEBUG("Start loading image in GPU");

  m_pitchCPU = inImageCPUPitch;
  m_inImageCPU = inImageCPU;

  checkCudaErrors(cudaMemcpy2D(m_inImageGPU, m_pitchInGPU, m_imageCPU, m_pitchCPU, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyHostToDevice));

  LOG_DEBUG("End loading image in GPU");
}

void CannyEdge<T>::unloadImage(unsigned char *outImageCPU)
{
  if (!outImageCPU)
  {
    LOG_ERROR("Cannot unload image from GPU");
    return;
  }

  LOG_DEBUG("Start unloading image from GPU");

  m_outImageCPU = outImageCPU;

  checkCudaErrors(cudaMemcpy2D(imageCPU, m_imageWidth * nbChannels, m_outImageGPU, m_pitchOutGPU, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyDeviceToHost));

  LOG_DEBUG("End unloading image in GPU");
}

bool CannyEdge<T>::readyToRun(unsigned int blockSize)
{
  if (!m_isAlloc)
  {
    LOG_ERROR("Cannot process if no memory is allocated on GPU");
    return false;
  }

  if (!m_imageCPU || m_pitchCPU == 0)
  {
    LOG_ERROR("Cannot process if no image is loaded from CPU");
    return false;
  }

  if (blockSize > m_imageWidth || blockSize > m_imageHeight)
  {
    LOG_ERROR("BlockSize cannot be bigger than imageSize");
    return false;
  }

  return true;
}

void CannyEdge<T>::runGaussianFilter(unsigned int blockSize)
{
  if (!CannyEdge::readyToRun(blockSize))
    return;

  LOG_DEBUG("Start Gaussian Filter Cuda Kernel");

  dim3 blocks(blockSize, blockSize, 1);
  dim3 grid((m_imageWidth * m_nbChannels) / blockSize, m_imageHeight / blockSize, 1);
  gaussianFilter2D_RGB<<<grid, blocks>>>(m_inImageGPU, m_outImageGPU, m_imageWidth * m_nbChannels, m_imageHeight, m_pitchInGPU, m_pitchOutGPU);

  LOG_DEBUG("End Gaussian Filter Cuda Kernel");
}

void CannyEdge<T>::runGradient(unsigned int blockSize){};
void CannyEdge<T>::runNonMaxSuppr(unsigned int blockSize){};
void CannyEdge<T>::runDoubleThresh(unsigned int blockSize){};
void CannyEdge<T>::runEdgeTracking(unsigned int blockSize){};
*/

//#endif
