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

    // Step 0 - Gray Conversion
    void runGrayConversion(unsigned int blockSize);
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

    unsigned int m_imageWidth;
    unsigned int m_imageHeight;
    unsigned int m_nbChannels;

    T *d_inRGB;
    size_t d_inPitch;

    T *d_outY;
    size_t d_outPitch;

    bool m_isAlloc;
  };

// Device Kernels
#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

  __constant__ float GaussianK[5][5];

  // Greyscale conversion
  // From 8UC3 to 8UC1
  __global__ void RGB2Y(const uint8_t *__restrict__ in, uint8_t *__restrict__ out, unsigned int width, unsigned int height, unsigned int pitchIn, unsigned int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      int iIn = row * pitchIn + 3 * col;
      out[row * pitchOut + col] = in[iIn] * 0.21f + in[iIn + 1] * 0.72f + in[iIn + 2] * 0.07f;
    }
  }

  // Gaussian Filter
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
    : m_isAlloc(false), m_nbChannels(nbChannels), m_imageWidth(imageWidth), m_imageHeight(imageHeight), d_inPitch(0), d_outPitch(0)
  {
    initAlloc();
  };
  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::~CannyEdge()
  {
    endAlloc();
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::loadImage(unsigned char *hImage, size_t hPitch)// Need to add checks
  {
    if (!hImage || hPitch == 0)
    {
      LOG_ERROR("Cannot load image to GPU");
      return;
    }

    LOG_DEBUG("Start loading image in GPU");

    checkCudaErrors(cudaMemcpy2D(d_inRGB, d_inPitch, hImage, hPitch, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyHostToDevice));

    LOG_DEBUG("End loading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::unloadImage(unsigned char *hImage)// Need to add checks
  {
    if (!hImage)
    {
      LOG_ERROR("Cannot unload image from GPU");
      return;
    }

    LOG_DEBUG("Start unloading image from GPU");

    checkCudaErrors(cudaMemcpy2D(hImage, m_imageWidth, d_outY, d_outPitch, m_imageWidth, m_imageHeight, cudaMemcpyDeviceToHost));

    LOG_DEBUG("End unloading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGrayConversion(unsigned int blockSize)
  {
    if (!CannyEdge::readyToRun(blockSize))
      return;

    LOG_DEBUG("Start Gray Conversion Cuda Kernel");

    dim3 blocks(blockSize, blockSize, 1);
    dim3 grid(m_imageWidth / blockSize + 1, m_imageHeight / blockSize + 1, 1);
    RGB2Y<<<grid, blocks>>>(d_inRGB, d_outY, m_imageWidth, m_imageHeight, d_inPitch, d_outPitch);

    LOG_DEBUG("End Gray Conversion Cuda Kernel");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGaussianFilter(unsigned int blockSize)
  {
    if (!CannyEdge::readyToRun(blockSize))
      return;

    LOG_DEBUG("Start Gaussian Filter Cuda Kernel");

    dim3 blocks(blockSize, blockSize, 1);
    dim3 grid((ceil(m_imageWidth * m_nbChannels) / blockSize), ceil(m_imageHeight / blockSize), 1);
    // gaussianFilter2D_RGB<<<grid, blocks>>>(d_inRGB, d_outY, m_imageWidth * m_nbChannels, m_imageHeight, d_inPitch, d_outPitch);

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

    checkCudaErrors(cudaMallocPitch(&d_inRGB, &d_inPitch, m_imageWidth * m_nbChannels, m_imageHeight));
    checkCudaErrors(cudaMallocPitch(&d_outY, &d_outPitch, m_imageWidth, m_imageHeight));

    //const std::array<std::array<float, 5>, 5> gaussianKernel = 1 / 159.0f * { { 2, 4, 5, 4, 2 }, { 4, 9, 12, 9, 4 }, { 5, 12, 15, 12, 5 }, { 4, 9, 12, 9, 4 }, { 2, 4, 5, 4, 2 } };
    //checkCudaErrors(cudaMemcpyToSymbol(GK, gaussianKernel, 25 * sizeof(float)));

    LOG_DEBUG("End allocating image memory in GPU");

    m_isAlloc = true;
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::endAlloc()
  {
    LOG_DEBUG("Start deallocating image memory in GPU");

    checkCudaErrors(cudaFree(d_inRGB));
    checkCudaErrors(cudaFree(d_outY));

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

    if (blockSize > m_imageWidth || blockSize > m_imageHeight)
    {
      LOG_ERROR("BlockSize cannot be bigger than imageSize");
      return false;
    }

    return true;
  }
}// namespace cuda
}// namespace cvp
