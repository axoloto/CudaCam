// Device Kernels

#include "cannyEdgeD.hpp"
#include <math_constants.h>

namespace cvp
{
namespace cuda
{
  // Gaussian kernel 5x5
  __constant__ float GK[5][5];

  // Mono
  constexpr float B_WEIGHT = 0.114f;
  constexpr float G_WEIGHT = 0.587f;
  constexpr float R_WEIGHT = 0.299f;
  constexpr int B_WT = static_cast<int>(64.0f * B_WEIGHT + 0.5f);
  constexpr int G_WT = static_cast<int>(64.0f * G_WEIGHT + 0.5f);
  constexpr int R_WT = static_cast<int>(64.0f * R_WEIGHT + 0.5f);

  // Gaussian
  constexpr int G_O_TILE_WIDTH = MAX_2D_BLOCK_SIDE - 4;// 4 halo cells

  // Sobel/NonMaxSuppr/Hysteresis
  constexpr int O_TILE_WIDTH = MAX_2D_BLOCK_SIDE - 2;// 2 halo cells

  // Grad mult coeff
  constexpr int GRAD_COEFF = 4;

  // DoubleTresh/NonMaxSuppr/Hysteresis
  constexpr unsigned char FINAL_EDGE = 255;
  constexpr unsigned char CANDIDATE_EDGE = 128;
  constexpr unsigned char NO_EDGE = 0;

  __global__ void float2uchar(
    const float *const __restrict__ in,
    unsigned char *const __restrict__ out,
    const int width,
    const int height,
    const int pitchIn,
    const int pitchOut)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      out[row * pitchOut + col] = (unsigned char)min(abs(in[row * pitchIn + col]), 255.0f);
    }
  }

  // Greyscale conversion
  __global__ void rgb2mono(
    const unsigned char *const __restrict__ rgb,
    unsigned char *const __restrict__ mono,
    const int width,
    const int height,
    const int pitchIn,
    const int pitchOut)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const int iRGB = row * pitchIn + 3 * col;
      mono[row * pitchOut + col] = min(255, (rgb[iRGB] * B_WT + rgb[iRGB + 1] * G_WT + rgb[iRGB + 2] * R_WT) >> 6);
    }
  }

  // Gaussian Filter
  __global__ void gaussianFilter5x5(
    const unsigned char *const __restrict__ mono,
    unsigned char *const __restrict__ blurr,
    const int width,
    const int height,
    const int monoPitch,
    const int blurrPitch)
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

  // Sobel X and Y
  __global__ void sobelXY(
    const unsigned char *const __restrict__ blurr,
    float *const __restrict__ sobelX,
    float *const __restrict__ sobelY,
    const int width,
    const int height,
    const int blurrPitch,
    const int sobelXPitch,
    const int sobelYPitch)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = O_TILE_WIDTH * blockIdx.y + ty;

    int row_i = row_o - 1;// maskWidth = 3/2
    int col_i = col_o - 1;// maskHeight = 3/2

    __shared__ unsigned char blurr_s[O_TILE_WIDTH + 2][O_TILE_WIDTH + 2];// 1 halo cell on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      blurr_s[ty][tx] = blurr[row_i * blurrPitch + col_i];
    }
    else
    {
      blurr_s[ty][tx] = 0;
    }

    __syncthreads();

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
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
    const int width,
    const int height,
    const int sobelXPitch,
    const int sobelYPitch,
    const int gradPitch,
    const int slopePitch)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const float sX = sobelX[row * sobelXPitch + col];
      const float sY = sobelY[row * sobelYPitch + col];

      grad[row * gradPitch + col] = GRAD_COEFF * sqrtf(sX * sX + sY * sY);
      slope[row * slopePitch + col] = atan2(sX, sY);
    }
  }

  // Non Maximum Suppression
  __global__ void nonMaxSuppr(
    const float *const __restrict__ grad,
    const float *const __restrict__ slope,
    unsigned char *const __restrict__ nms,
    const int width,
    const int height,
    const int gradPitch,
    const int slopePitch,
    const int nmsPitch)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = O_TILE_WIDTH * blockIdx.y + ty;

    int row_i = row_o - 1;
    int col_i = col_o - 1;

    __shared__ float grad_s[O_TILE_WIDTH + 2][O_TILE_WIDTH + 2];// 1 halo cells on each side of the tile

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      grad_s[ty][tx] = grad[row_i * gradPitch + col_i];
    }
    else
    {
      grad_s[ty][tx] = 0;
    }

    __syncthreads();

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
    {
      if (col_o < width && row_o < height)
      {
        const float gradVal = grad_s[ty + 1][tx + 1];

        float angle = slope[row_o * slopePitch + col_o] * 180.0f / CUDART_PI_F;
        if (angle < 0.0f) angle += 180.0f;

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

        // Suppressing values wich are not local maximums
        nms[row_o * nmsPitch + col_o] = (q <= gradVal && r <= gradVal) ? min((unsigned char)gradVal, 255) : 0;
      }
    }
  }

  // Double Threshold, finding Final Edges and Candidate Edges
  __global__ void doubleThreshold(
    const unsigned char *const __restrict__ nms,
    unsigned char *const __restrict__ thresh,
    const int width,
    const int height,
    const int nmsPitch,
    const int threshPitch,
    const unsigned char lowThreshold,
    const unsigned char highThreshold)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const unsigned char nmsVal = nms[row * nmsPitch + col];

      thresh[row * threshPitch + col] = nmsVal > highThreshold ? FINAL_EDGE : nmsVal > lowThreshold ? CANDIDATE_EDGE
                                                                                                    : NO_EDGE;
    }
  }

  __global__ void hysteresis(
    const unsigned char *const __restrict__ thresh,
    unsigned char *const __restrict__ hyster,
    int *const __restrict__ isImageModified,
    const int width,
    const int height,
    const int threshPitch,
    const int hysterPitch)
  {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    int row_o = O_TILE_WIDTH * blockIdx.y + ty;

    int row_i = row_o - 1;
    int col_i = col_o - 1;

    __shared__ int isBlockModified[2];
    __shared__ float thresh_s[O_TILE_WIDTH + 2][O_TILE_WIDTH + 2];// 1 halo cell on each side of the tile

    if (tx == 0 && ty == 0)
    {
      isBlockModified[0] = 1;//Local modification alert, only at internal interation level
      isBlockModified[1] = 0;//Global modification alert, to ask host for another kernel runtime
    }

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
      thresh_s[ty][tx] = thresh[row_i * threshPitch + col_i];
    }
    else
    {
      thresh_s[ty][tx] = 0;
    }

    __syncthreads();

    while (isBlockModified[0])
    {
      if (tx == 0 && ty == 0)
        isBlockModified[0] = 0;

      __syncthreads();

      if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
      {
        if (thresh_s[ty + 1][tx + 1] == CANDIDATE_EDGE
            && (thresh_s[ty + 2][tx + 2] == FINAL_EDGE
                || thresh_s[ty + 2][tx + 1] == FINAL_EDGE
                || thresh_s[ty + 2][tx] == FINAL_EDGE
                || thresh_s[ty][tx + 2] == FINAL_EDGE
                || thresh_s[ty][tx + 1] == FINAL_EDGE
                || thresh_s[ty][tx] == FINAL_EDGE
                || thresh_s[ty + 1][tx] == FINAL_EDGE
                || thresh_s[ty + 1][tx + 2] == FINAL_EDGE))
        {
          thresh_s[ty + 1][tx + 1] = FINAL_EDGE;

          if (!isBlockModified[0])
            isBlockModified[0] = 1;
        }
      }

      __syncthreads();

      if (tx == 0 && ty == 0 && isBlockModified[0])
        isBlockModified[1] = 1;
    }

    __syncthreads();

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
    {
      if (col_o < width && row_o < height)
      {
        hyster[row_o * hysterPitch + col_o] = thresh_s[ty + 1][tx + 1];
      }
    }

    if (tx == 0 && ty == 0 && isBlockModified[1])
      atomicAdd(&isImageModified[0], 1);
  }

  __global__ void removeCandidates(
    const unsigned char *const __restrict__ in,
    unsigned char *const __restrict__ out,
    const int width,
    const int height,
    const int inPitch,
    const int outPitch)
  {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
      const unsigned int inVal = in[row * inPitch + col];
      out[row * outPitch + col] = (inVal == CANDIDATE_EDGE) ? 0 : inVal;
    }
  }

}// namespace cuda
}// namespace cvp