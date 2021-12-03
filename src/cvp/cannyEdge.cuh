#pragma once

#include <stdio.h>
#include <stdint.h>
#include "logging.hpp"
#include "helper.cuh"
namespace cvp
{
namespace cuda
{
  // Edge detection operator using multi-stage algorithm to detect edges in images invented by John F. Canny
  // A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 8, 1986, pp. 679�698.
  template<class T, size_t nbChannels>
  class CannyEdge
  {
  public:
    CannyEdge(unsigned int imageWidth, unsigned int imageHeight);
    ~CannyEdge();

    void loadImage(unsigned char *ptrCPU, size_t pitchCPU);
    void unloadImage(unsigned char *ptrCPU);

    // Step 1 - Noise Reduction
    void runGaussianFilter(unsigned int blockSize);
    // Step 2 - Gradient Calculation
    void runGradient(unsigned int blockSize);
    // Step 3 - Non-maximum suppression
    void runNonMaxSuppr(unsigned int blockSize);
    // Step 4 - Double threshold
    void runDoubleThresh(unsigned int blockSize);
    // Step 5 - Edge Tracking by hysteresis
    void runEdgeTracking(unsigned int blockSize);

  protected:
    void initAlloc();
    void endAlloc();

    bool readyToRun(unsigned int blockSize);

    // CPU data
    T *m_inImageCPU;
    T *m_outImageCPU;
    unsigned int m_imageWidth;
    unsigned int m_imageHeight;
    unsigned int m_nbChannels;
    size_t m_pitchCPU;

    // GPU data
    T *m_inImageGPU;
    T *m_outImageGPU;
    size_t m_pitchInGPU;
    size_t m_pitchOutGPU;

    bool m_isAlloc;
  };

// Device Kernels
#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

  __constant__ float GaussianK[5][5];

  __global__ void gaussianFilter2D_RGB(uint8_t *in, uint8_t *out, unsigned int width, unsigned int height, unsigned int pitchIn, unsigned int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < height && col < width)
      out[row * pitchOut + col] = in[row * pitchIn + col] + 3;
  }
#endif

  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::CannyEdge(unsigned int imageWidth, unsigned int imageHeight)
    : m_isAlloc(false), m_nbChannels(nbChannels), m_imageWidth(imageWidth), m_imageHeight(imageHeight), m_pitchInGPU(0), m_pitchOutGPU(0)
  {
    initAlloc();
  };
  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::~CannyEdge()
  {
    endAlloc();
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::loadImage(unsigned char *ptrCPU, size_t pitchCPU)
  {
    if (!ptrCPU)
    {
      LOG_ERROR("Cannot load image to GPU");
      return;
    }

    LOG_DEBUG("Start loading image in GPU");

    m_pitchCPU = pitchCPU;
    m_inImageCPU = ptrCPU;

    checkCudaErrors(cudaMemcpy2D(m_inImageGPU, m_pitchInGPU, m_inImageCPU, m_pitchCPU, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyHostToDevice));

    LOG_DEBUG("End loading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::unloadImage(unsigned char *ptrCPU)
  {
    if (!ptrCPU)
    {
      LOG_ERROR("Cannot unload image from GPU");
      return;
    }

    LOG_DEBUG("Start unloading image from GPU");

    m_outImageCPU = ptrCPU;

    checkCudaErrors(cudaMemcpy2D(m_outImageCPU, m_imageWidth * nbChannels, m_outImageGPU, m_pitchOutGPU, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyDeviceToHost));

    LOG_DEBUG("End unloading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGaussianFilter(unsigned int blockSize)
  {
    if (!CannyEdge::readyToRun(blockSize))
      return;

    LOG_DEBUG("Start Gaussian Filter Cuda Kernel");

    dim3 blocks(blockSize, blockSize, 1);
    dim3 grid((m_imageWidth * m_nbChannels) / blockSize, m_imageHeight / blockSize, 1);
    gaussianFilter2D_RGB<<<grid, blocks>>>(m_inImageGPU, m_outImageGPU, m_imageWidth * m_nbChannels, m_imageHeight, m_pitchInGPU, m_pitchOutGPU);

    LOG_DEBUG("End Gaussian Filter Cuda Kernel");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGradient(unsigned int blockSize){};
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runNonMaxSuppr(unsigned int blockSize){};
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runDoubleThresh(unsigned int blockSize){};
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runEdgeTracking(unsigned int blockSize){};

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::initAlloc()
  {
    LOG_DEBUG("Start allocating image memory in GPU");

    checkCudaErrors(cudaMallocPitch(&m_inImageGPU, &m_pitchInGPU, m_imageWidth * m_nbChannels, m_imageHeight));
    checkCudaErrors(cudaMallocPitch(&m_outImageGPU, &m_pitchOutGPU, m_imageWidth * m_nbChannels, m_imageHeight));

    LOG_DEBUG("End allocating image memory in GPU");

    m_isAlloc = true;
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::endAlloc()
  {
    LOG_DEBUG("Start deallocating image memory in GPU");

    checkCudaErrors(cudaFree(m_inImageGPU));
    checkCudaErrors(cudaFree(m_outImageGPU));

    LOG_DEBUG("End deallocating image memory in GPU");

    m_isAlloc = false;
  }

  template<class T, size_t nbChannels>
  bool CannyEdge<T, nbChannels>::readyToRun(unsigned int blockSize)
  {
    if (!m_isAlloc)
    {
      LOG_ERROR("Cannot process if no memory is allocated on GPU");
      return false;
    }

    if (!m_inImageCPU || m_pitchCPU == 0)
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
}// namespace cuda
}// namespace cvp
