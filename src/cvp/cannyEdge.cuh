#pragma once

#ifdef _WIN32// Necessary for OpenGL
#include <windows.h>
#endif

#include <stdio.h>
#include <stdint.h>
#include <opencv2/core.hpp>
#include <cuda_gl_interop.h>

#include "logging.hpp"
#include "define.hpp"
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

    void unloadImage(T *ptrCPU);// Not needed anymore, using interop opengl cuda

    void run(cv::Mat input);

  private:
    void _initAlloc();
    void _endAlloc();

    void _loadInputImage(T *ptrCPU, size_t pitchCPU);
    // Step 0 - Gray Conversion
    void _runGrayConversion();
    // Step 1 - Noise Reduction
    void _runGaussianFilter();
    // Step 2 - Gradient Calculation
    void _runGradient();
    // Step 3 - Non-maximum suppression
    void _runNonMaxSuppr();
    // Step 4 - Double threshold
    void _runDoubleThresh();
    // Step 5 - Edge Tracking by hysteresis
    void runEdgeTracking();

    unsigned int m_imageWidth;
    unsigned int m_imageHeight;
    unsigned int m_nbChannels;

    int m_inputBlockSize;

    T *d_RGB;
    size_t d_inPitch;

    unsigned char *d_mono;
    size_t d_monoPitch;

    unsigned char *d_blurr;
    size_t d_blurrPitch;

    float *d_sobelX;
    size_t d_sobelXPitch;

    float *d_sobelY;
    size_t d_sobelYPitch;

    float *d_grad;
    size_t d_gradPitch;

    float *d_slope;
    size_t d_slopePitch;

    unsigned char *d_nms;
    size_t d_nmsPitch;

    unsigned char *d_thresh;
    size_t d_threshPitch;

    struct cudaGraphicsResource *d_pbo;

    bool m_isAlloc;
    bool m_isInit;
  };

// Device Kernels
#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

#include <math_constants.h>

  __global__ void TEST(
    const unsigned char *const __restrict__ in,
    unsigned char *const __restrict__ out,
    const unsigned int width,
    const unsigned int height,
    const unsigned int pitchIn,
    const unsigned int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      out[row * pitchOut + col] = in[row * pitchIn + col];
    }
  }

  constexpr float B_WEIGHT = 0.114f;
  constexpr float G_WEIGHT = 0.587f;
  constexpr float R_WEIGHT = 0.299f;

  constexpr int B_WT = static_cast<int>(64.0f * B_WEIGHT + 0.5f);
  constexpr int G_WT = static_cast<int>(64.0f * G_WEIGHT + 0.5f);
  constexpr int R_WT = static_cast<int>(64.0f * R_WEIGHT + 0.5f);

  // Greyscale conversion
  __global__ void RGB2Mono(
    const unsigned char *const __restrict__ rgb,
    unsigned char *const __restrict__ mono,
    const unsigned int width,
    const unsigned int height,
    const unsigned int pitchIn,
    const unsigned int pitchOut)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const int iRGB = row * pitchIn + 3 * col;
      mono[row * pitchOut + col] = min(255, (rgb[iRGB] * B_WT + rgb[iRGB + 1] * G_WT + rgb[iRGB + 2] * R_WT) >> 6);
    }
  }

  constexpr int G_O_TILE_WIDTH = 28;// Must be -4 blockSize

  __constant__ float GK[5][5];

  // Gaussian Filter
  __global__ void gaussianFilter5x5(
    const unsigned char *const __restrict__ mono,
    unsigned char *const __restrict__ blurr,
    const unsigned int width,
    const unsigned int height,
    const unsigned int monoPitch,
    const unsigned int blurrPitch)
  {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col_o = G_O_TILE_WIDTH * blockIdx.x + tx;
    const int row_o = G_O_TILE_WIDTH * blockIdx.y + ty;

    const int row_i = row_o - 2;// maskWidth  = 5/2
    const int col_i = col_o - 2;// maskHeight = 5/2

    __shared__ unsigned char mono_s[G_O_TILE_WIDTH + 4][G_O_TILE_WIDTH + 4];// 2 halo cells on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      mono_s[ty][tx] = mono[row_i * monoPitch + col_i];
    }
    else
    {
      mono_s[ty][tx] = 0;
    }

    __syncthreads();

    float fSum = 0.0f;
    if (ty < G_O_TILE_WIDTH && tx < G_O_TILE_WIDTH)
    {
      for (int r = 0; r < 5; ++r)
      {
        for (int c = 0; c < 5; ++c)
        {
          fSum += GK[r][c] * (float)(mono_s[ty + r][tx + c]);
        }
      }

      if (col_o < width && row_o < height)
      {
        blurr[row_o * blurrPitch + col_o] = (unsigned char)fSum;
      }
    }
  }

  constexpr int S_O_TILE_WIDTH = 30;// Must be -2 blockSize

  // Sobel X and Y
  __global__ void sobelXY(
    const unsigned char *const __restrict__ blurr,
    float *const __restrict__ sobelX,
    float *const __restrict__ sobelY,
    const unsigned int width,
    const unsigned int height,
    const unsigned int blurrPitch,
    const unsigned int sobelXPitch,
    const unsigned int sobelYPitch)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = S_O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = S_O_TILE_WIDTH * blockIdx.y + ty;

    int row_i = row_o - 1;// maskWidth = 3/2
    int col_i = col_o - 1;// maskHeight = 3/2

    __shared__ unsigned char blurr_s[S_O_TILE_WIDTH + 2][S_O_TILE_WIDTH + 2];// 1 halo cells on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      blurr_s[ty][tx] = blurr[row_i * blurrPitch + col_i];
    }
    else
    {
      blurr_s[ty][tx] = 0;
    }

    __syncthreads();

    if (ty < S_O_TILE_WIDTH && tx < S_O_TILE_WIDTH)
    {
      if (col_o < width && row_o < height)
      {
        // shared mem is shifted +1
        int sumX = 0;
        sumX += -blurr_s[ty][tx] + blurr_s[ty][tx + 2];
        sumX += -2 * blurr_s[ty + 1][tx] + 2 * blurr_s[ty + 1][tx + 2];
        sumX += -blurr_s[ty + 2][tx] + blurr_s[ty + 2][tx + 2];

        sobelX[row_o * sobelXPitch + col_o] = (float)(sumX) / 8.0f;

        int sumY = 0;
        sumY += blurr_s[ty][tx] + 2 * blurr_s[ty][tx + 1] + blurr_s[ty][tx + 2];
        sumY -= blurr_s[ty + 2][tx] + 2 * blurr_s[ty + 2][tx + 1] + blurr_s[ty + 2][tx + 2];

        sobelY[row_o * sobelYPitch + col_o] = (float)(sumY) / 8.0f;
      }
    }
  }

  // Grad and Slope
  __global__ void gradSlope(
    const float *const __restrict__ sobelX,
    const float *const __restrict__ sobelY,
    float *const __restrict__ grad,
    float *const __restrict__ slope,
    const unsigned int width,
    const unsigned int height,
    const unsigned int sobelXPitch,
    const unsigned int sobelYPitch,
    const unsigned int gradPitch,
    const unsigned int slopePitch)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const float sX = sobelX[row * sobelXPitch + col];
      const float sY = sobelY[row * sobelYPitch + col];

      grad[row * gradPitch + col] = 4 * sqrtf(sX * sX + sY * sY);
      slope[row * slopePitch + col] = atan2(sX, sY);
    }
  }

  constexpr int N_O_TILE_WIDTH = 30;// Must be -2 blockSize

  // Non Maximum Suppression
  __global__ void nonMaxSuppr(
    const float *const __restrict__ grad,
    const float *const __restrict__ slope,
    unsigned char *const __restrict__ nms,
    const unsigned int width,
    const unsigned int height,
    const unsigned int gradPitch,
    const unsigned int slopePitch,
    const unsigned int nmsPitch)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = N_O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = N_O_TILE_WIDTH * blockIdx.y + ty;

    int row_i = row_o - 1;
    int col_i = col_o - 1;

    __shared__ float grad_s[N_O_TILE_WIDTH + 2][N_O_TILE_WIDTH + 2];// 1 halo cells on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      grad_s[ty][tx] = grad[row_i * gradPitch + col_i];
    }
    else
    {
      grad_s[ty][tx] = 0;
    }

    __syncthreads();

    if (ty < N_O_TILE_WIDTH && tx < N_O_TILE_WIDTH)
    {
      if (col_o < width && row_o < height)
      {
        const float gradVal = grad_s[ty + 1][tx + 1];
        float angle = slope[row_o * slopePitch + col_o] * 180.0f / CUDART_PI_F;
        if (angle < 0.0f) angle += 180.0f;

        /*
        // clang-format off
        // Identify edge direction based on the angle value and retrieve grad values for this direction
        uchar2 qr =  (angle < 22.5f || angle > 157.5f) ? make_uchar2(grad_s[ty][tx + 1], grad_s[ty][tx - 1])
                  : ((22.5f <= angle && angle <= 67.5f) ? make_uchar2(grad_s[ty + 1][tx - 1], grad_s[ty - 1][tx + 1])
                  : ((67.5f < angle && angle <= 112.5f) ? make_uchar2(grad_s[ty + 1][tx], grad_s[ty - 1][tx])
                  : ((112.5f < angle && angle <= 157.5f) ? make_uchar2(grad_s[ty - 1][tx - 1], grad_s[ty + 1][tx + 1])
                  : make_uchar2(0, 0))));
        // clang-format on    
        */
        float q = 0.0f;
        float r = 0.0f;

        if (angle < 22.5f || angle > 157.5f)
        {
          q = grad_s[ty + 2][tx + 1];
          r = grad_s[ty][tx + 1];
        }
        else if (22.5f <= angle && angle <= 67.5f)
        {
          q = grad_s[ty + 2][tx];
          r = grad_s[ty][tx + 2];
        }
        else if (67.5f < angle && angle <= 112.5f)
        {
          q = grad_s[ty + 1][tx + 2];
          r = grad_s[ty + 1][tx];
        }
        else if (112.5f < angle && angle <= 157.5f)
        {
          q = grad_s[ty][tx];
          r = grad_s[ty + 2][tx + 2];
        }

        // Suppressing values wich are not maximums
        nms[row_o * nmsPitch + col_o] = (q <= gradVal && r <= gradVal) ? min((unsigned char)gradVal, 255) : 0;
      }
    }
  }

  // Double Threshold, finding Final Edges and Candidate Edges
  __global__ void doubleThreshold(
    const unsigned char *const __restrict__ nms,
    unsigned char *const __restrict__ thresh,
    const unsigned int width,
    const unsigned int height,
    const unsigned int nmsPitch,
    const unsigned int threshPitch)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const unsigned char nmsVal = nms[row * nmsPitch + col];

      thresh[row * threshPitch + col] = nmsVal > 80 ? 255 : nmsVal > 15 ? 25
                                                                        : 0;
    }
  }

#endif

  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::CannyEdge(unsigned int pbo, unsigned int imageWidth, unsigned int imageHeight)
    : m_isAlloc(false), m_nbChannels(nbChannels), m_imageWidth(imageWidth), m_imageHeight(imageHeight), m_inputBlockSize(32), d_inPitch(0), d_monoPitch(0)
  {
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&d_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));
    _initAlloc();
  };

  template<class T, size_t nbChannels>
  CannyEdge<T, nbChannels>::~CannyEdge()
  {
    _endAlloc();
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::run(cv::Mat input)// Need to add checks
  {
    if (!m_isAlloc)
    {
      LOG_ERROR("Cannot Process Canny Edge Filter, missing allocation");
    }

    LOG_DEBUG("Start Canny Edge Filter on CUDA device");

    _loadInputImage(input.ptr(), input.step);
    _runGrayConversion();
    _runGaussianFilter();
    _runGradient();
    _runNonMaxSuppr();
    _runDoubleThresh();

    LOG_DEBUG("End Canny Edge Filter on CUDA device");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_loadInputImage(T *hImage, size_t hPitch)// Need to add checks
  {
    if (!hImage || hPitch == 0)
    {
      LOG_ERROR("Cannot load image to GPU");
      return;
    }

    LOG_DEBUG("Start loading image in GPU");

    checkCudaErrors(cudaMemcpy2D(d_RGB, d_inPitch, hImage, hPitch, m_imageWidth * m_nbChannels, m_imageHeight, cudaMemcpyHostToDevice));

    LOG_DEBUG("End loading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::unloadImage(T *hImage)
  {
    if (!hImage)
    {
      LOG_ERROR("Cannot unload image from GPU");
      return;
    }

    LOG_DEBUG("Start unloading image from GPU");

    checkCudaErrors(cudaMemcpy2D(hImage, m_imageWidth, d_blurr, d_blurrPitch, m_imageWidth, m_imageHeight, cudaMemcpyDeviceToHost));

    LOG_DEBUG("End unloading image in GPU");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runGrayConversion()
  {
    LOG_DEBUG("Start Gray Conversion");

    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_imageWidth + m_inputBlockSize - 1) / m_inputBlockSize, (m_imageHeight + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    RGB2Mono<<<grid, blocks>>>(d_RGB, d_mono, m_imageWidth, m_imageHeight, d_inPitch, d_monoPitch);

    LOG_DEBUG("End Gray Conversion");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runGaussianFilter()
  {
    LOG_DEBUG("Start Gaussian Filter");

    int outputBlockSize = m_inputBlockSize - 4;//2 halo cells on each border
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_imageWidth + outputBlockSize - 1) / outputBlockSize, (m_imageHeight + outputBlockSize - 1) / outputBlockSize, 1);
    gaussianFilter5x5<<<grid, blocks>>>(d_mono, d_blurr, m_imageWidth, m_imageHeight, d_monoPitch, d_blurrPitch);

    LOG_DEBUG("End Gaussian Filter");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runGradient()
  {
    LOG_DEBUG("Start Gradient Computation");
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);

    LOG_DEBUG(" 1-Sobel");
    int outputBlockSize = m_inputBlockSize - 2;//1 halo cell on each border
    dim3 sobelGrid((m_imageWidth + outputBlockSize - 1) / outputBlockSize, (m_imageHeight + outputBlockSize - 1) / outputBlockSize, 1);
    sobelXY<<<sobelGrid, blocks>>>(d_blurr, d_sobelX, d_sobelY, m_imageWidth, m_imageHeight, d_blurrPitch, d_sobelXPitch / sizeof(float), d_sobelYPitch / sizeof(float));

    LOG_DEBUG(" 2-Gradient");
    dim3 gradGrid((m_imageWidth + m_inputBlockSize - 1) / m_inputBlockSize, (m_imageHeight + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    gradSlope<<<gradGrid, blocks>>>(d_sobelX, d_sobelY, d_grad, d_slope, m_imageWidth, m_imageHeight, d_sobelXPitch / sizeof(float), d_sobelYPitch / sizeof(float), d_gradPitch / sizeof(float), d_slopePitch / sizeof(float));

    LOG_DEBUG("End Gradient Computation");
  }

  /*
  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runNonMaxSuppr()
  {
    LOG_DEBUG("Start Non Max Suppression");

    size_t dumbSize;
    unsigned char *outPbo;
    checkCudaErrors(cudaGraphicsMapResources(1, &d_pbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&outPbo, &dumbSize, d_pbo));

    int outputBlockSize = m_inputBlockSize - 2;
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_imageWidth + outputBlockSize - 1) / outputBlockSize, (m_imageHeight + outputBlockSize - 1) / outputBlockSize, 1);
    nonMaxSuppr<<<grid, blocks>>>(d_grad, d_slope, outPbo, m_imageWidth, m_imageHeight, d_gradPitch / sizeof(float), d_slopePitch / sizeof(float), m_imageWidth);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &d_pbo, 0));

    LOG_DEBUG("End Non Max Suppression");
  }*/

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runNonMaxSuppr()
  {
    LOG_DEBUG("Start Non Max Suppression");

    int outputBlockSize = m_inputBlockSize - 2;
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_imageWidth + outputBlockSize - 1) / outputBlockSize, (m_imageHeight + outputBlockSize - 1) / outputBlockSize, 1);
    nonMaxSuppr<<<grid, blocks>>>(d_grad, d_slope, d_nms, m_imageWidth, m_imageHeight, d_gradPitch / sizeof(float), d_slopePitch / sizeof(float), d_nmsPitch);

    LOG_DEBUG("End Non Max Suppression");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_runDoubleThresh()
  {
    LOG_DEBUG("Start Double Threshold");

    size_t dumbSize;
    unsigned char *outPbo;
    checkCudaErrors(cudaGraphicsMapResources(1, &d_pbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&outPbo, &dumbSize, d_pbo));

    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_imageWidth + m_inputBlockSize - 1) / m_inputBlockSize, (m_imageHeight + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    doubleThreshold<<<grid, blocks>>>(d_nms, outPbo, m_imageWidth, m_imageHeight, d_nmsPitch, m_imageWidth);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &d_pbo, 0));

    LOG_DEBUG("End Double Threshold");
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::runEdgeTracking(){};

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_initAlloc()
  {
    LOG_DEBUG("Start allocating image memory in GPU");

    checkCudaErrors(cudaMallocPitch(&d_RGB, &d_inPitch, m_imageWidth * m_nbChannels, m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_mono, &d_monoPitch, m_imageWidth, m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_blurr, &d_blurrPitch, m_imageWidth, m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_sobelX, &d_sobelXPitch, m_imageWidth * sizeof(float), m_imageHeight));
    checkCudaErrors(cudaMallocPitch(&d_sobelY, &d_sobelYPitch, m_imageWidth * sizeof(float), m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_grad, &d_gradPitch, m_imageWidth * sizeof(float), m_imageHeight));
    checkCudaErrors(cudaMallocPitch(&d_slope, &d_slopePitch, m_imageWidth * sizeof(float), m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_nms, &d_nmsPitch, m_imageWidth, m_imageHeight));

    checkCudaErrors(cudaMallocPitch(&d_thresh, &d_threshPitch, m_imageWidth, m_imageHeight));

    std::array<std::array<float, 5>, 5> GK_CPU = { { { 2, 4, 5, 4, 2 }, { 4, 9, 12, 9, 4 }, { 5, 12, 15, 12, 5 }, { 4, 9, 12, 9, 4 }, { 2, 4, 5, 4, 2 } } };
    for (int i = 0; i < 5; ++i)
    {
      for (int j = 0; j < 5; ++j)
      {
        GK_CPU[i][j] *= 1 / 159.0f;
      }
    }
    checkCudaErrors(cudaMemcpyToSymbol(GK, GK_CPU.data(), 25 * sizeof(float)));

    LOG_DEBUG("End allocating image memory in GPU");

    m_isAlloc = true;
  }

  template<class T, size_t nbChannels>
  void CannyEdge<T, nbChannels>::_endAlloc()
  {
    LOG_DEBUG("Start deallocating image memory in GPU");

    checkCudaErrors(cudaFree(d_RGB));
    checkCudaErrors(cudaFree(d_mono));
    checkCudaErrors(cudaFree(d_blurr));
    checkCudaErrors(cudaFree(d_sobelX));
    checkCudaErrors(cudaFree(d_sobelY));
    checkCudaErrors(cudaFree(d_grad));
    checkCudaErrors(cudaFree(d_slope));
    checkCudaErrors(cudaFree(d_nms));
    checkCudaErrors(cudaFree(d_thresh));

    LOG_DEBUG("End deallocating image memory in GPU");

    m_isAlloc = false;
  }
}// namespace cuda
}// namespace cvp
