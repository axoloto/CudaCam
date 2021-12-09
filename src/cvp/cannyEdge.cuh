#pragma once

#ifdef _WIN32// Necessary for OpenGL
#include <windows.h>
#endif

#include <stdio.h>
#include <stdint.h>

#include <cuda_gl_interop.h>

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
    CannyEdge(unsigned int pbo, unsigned int imageWidth, unsigned int imageHeight);
    ~CannyEdge();

    void loadInputImage(unsigned char *ptrCPU, size_t pitchCPU);
    //   void unloadImage(unsigned char *ptrCPU); // Not needed anymore, using interop opengl cuda

    // Step 0 - Gray Conversion
    void runGrayConversion(unsigned int blockSize);
    // Step 1 - Noise Reduction
    void runGaussianFilter();
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
    size_t d_inRGBPitch;

    unsigned char *d_inMono;
    size_t d_inMonoPitch;

    unsigned char *d_outY;
    size_t d_outPitch;

    struct cudaGraphicsResource *d_pbo;

    bool m_isAlloc;
    bool m_isInit;
  };

// Device Kernels
#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

  __constant__ float GK[5][5];

  // Greyscale conversion
  // From 8UC3 to 8UC1
  __global__ void RGB2Mono(const unsigned char *__restrict__ in, unsigned char *__restrict__ out, const unsigned int width, const unsigned int height, const unsigned int pitchIn, const unsigned int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      int iIn = row * pitchIn + 3 * col;
      out[row * pitchOut + col] = in[iIn] * 0.21f + in[iIn + 1] * 0.72f + in[iIn + 2] * 0.07f;
    }
  }

  __global__ void TEST(const unsigned char *__restrict__ in, unsigned char *__restrict__ out, const unsigned int width, const unsigned int height, const unsigned int pitchIn, const unsigned int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      out[row * pitchOut + col] = 130;//in[row * pitchIn + col];
    }
  }

#define O_TILE_WIDTH 30// Must be -4 blockSize

  // Gaussian Filter
  __global__ void gaussianFilter5x5(const unsigned char *__restrict__ in, unsigned char *out, const unsigned int width, const unsigned int height, const unsigned int pitchIn, const unsigned int pitchOut)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = O_TILE_WIDTH * blockIdx.y + ty;
    //int col_o = blockDim.x * blockIdx.x + tx;
    //int row_o = blockDim.y * blockIdx.y + ty;

    //if (col_o < width && row_o < height)
    //  {
    //  out[row_o * pitchOut + col_o] = 50;// (unsigned char)fval;
    // }
    /*
    int row_i = row_o - 2;// maskWidth = 5/2
    int col_i = col_o - 2;// maskHeight = 5/2

    __shared__ unsigned char in_s[O_TILE_WIDTH + 4][O_TILE_WIDTH + 4];// 2 halo cells on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      in_s[ty][tx] = in[row_i * pitchIn + col_i];
    }
    else
    {
      in_s[ty][tx] = 0;
    }

    __syncthreads();
    */
    float fval = 0.0f;
    if (ty < (O_TILE_WIDTH) && tx < (O_TILE_WIDTH))
    {
      /*
      for (int r = 0; r < 5; ++r)
      {
        for (int c = 0; c < 5; ++c)
        {
          fval += GK[r][c] * (float)(in_s[ty + r][tx + c]);
        }
      }
      */
      if (col_o < width && row_o < height)
      {
        out[row_o * pitchOut + col_o] = 30;// (unsigned char)fval;
      }
    }
  }
#endif

  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::CannyEdge(unsigned int pbo, unsigned int imageWidth, unsigned int imageHeight)
    : m_isAlloc(false), m_nbChannels(nbChannels), m_imageWidth(imageWidth), m_imageHeight(imageHeight), d_inRGBPitch(0), d_inMonoPitch(0)
  {
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&d_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));
    initAlloc();
  };

  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::~CannyEdge()
  {
    endAlloc();
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::loadInputImage(unsigned char *hImage, size_t hPitch)// Need to add checks
  {
    if (!hImage || hPitch == 0)
    {
      LOG_ERROR("Cannot load image to GPU");
      return;
    }

    LOG_DEBUG("Start loading image in GPU");

    checkCudaErrors(cudaMemcpy2D(d_inRGB, d_inRGBPitch, hImage, hPitch, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyHostToDevice));

    LOG_DEBUG("End loading image in GPU");
  }
  /*
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::unloadImage(unsigned char *hImage)
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
  */
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGrayConversion(unsigned int blockSize)
  {
    if (!CannyEdge::readyToRun(blockSize))
      return;

    LOG_DEBUG("Start Gray Conversion Cuda Kernel");
    //  unsigned char *outPbo;
    // size_t dumbSize;
    //checkCudaErrors(cudaGraphicsMapResources(1, &d_pbo, 0));
    // checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&outPbo, &dumbSize, d_pbo));

    dim3 blocks(blockSize, blockSize, 1);
    dim3 grid((m_imageWidth + blockSize - 1) / blockSize, (m_imageHeight + blockSize - 1) / blockSize, 1);
    RGB2Mono<<<grid, blocks>>>(d_inRGB, d_inMono, m_imageWidth, m_imageHeight, d_inRGBPitch, d_inMonoPitch);//d_outPitch
    //RGB2Mono<<<grid, blocks>>>(d_inRGB, outPbo, m_imageWidth, m_imageHeight, d_inRGBPitch, m_imageWidth);//d_outPitch

    //checkCudaErrors(cudaGraphicsUnmapResources(1, &d_pbo, 0));

    LOG_DEBUG("End Gray Conversion Cuda Kernel");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runGaussianFilter()
  {
    if (!CannyEdge::readyToRun(34))
      return;

    LOG_DEBUG("Start Gaussian Filter Cuda Kernel");

    unsigned char *outPbo;
    size_t dumbSize;
    checkCudaErrors(cudaGraphicsMapResources(1, &d_pbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&outPbo, &dumbSize, d_pbo));

    int blockSize = 30;// WIP Hardcoded in kernel, 30 + 2 + 2
    dim3 blocks(blockSize, blockSize, 1);
    dim3 grid((m_imageWidth + blockSize - 1) / blockSize, (m_imageHeight + blockSize - 1) / blockSize, 1);
    gaussianFilter5x5<<<grid, blocks>>>(d_inMono, outPbo, m_imageWidth, m_imageHeight, d_inMonoPitch, m_imageWidth);
    //TEST<<<grid, blocks>>>(d_inMono, outPbo, m_imageWidth, m_imageHeight, d_inMonoPitch, m_imageWidth);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &d_pbo, 0));

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

    checkCudaErrors(cudaMallocPitch(&d_inRGB, &d_inRGBPitch, m_imageWidth * m_nbChannels, m_imageHeight));
    checkCudaErrors(cudaMallocPitch(&d_inMono, &d_inMonoPitch, m_imageWidth, m_imageHeight));
    //
    std::array<std::array<float, 5>, 5> GK_CPU = { { { 2, 4, 5, 4, 2 }, { 4, 9, 12, 9, 4 }, { 5, 12, 15, 12, 5 }, { 4, 9, 12, 9, 4 }, { 2, 4, 5, 4, 2 } } };
    //GK_CPU *= 1 / 159.0f;
    for (int i = 0; i < 5; ++i)
    {
      for (int j = 0; j < 5; ++j)
      {
        //     GK_CPU[i][j] *= 1 / 159.0f;
      }
    }
    checkCudaErrors(cudaMemcpyToSymbol(GK, GK_CPU.data(), 25 * sizeof(float)));

    LOG_DEBUG("End allocating image memory in GPU");

    m_isAlloc = true;
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::endAlloc()
  {
    LOG_DEBUG("Start deallocating image memory in GPU");

    checkCudaErrors(cudaFree(d_inRGB));
    checkCudaErrors(cudaFree(d_inMono));

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
