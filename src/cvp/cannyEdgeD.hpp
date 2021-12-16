#pragma once

namespace cvp
{
namespace cuda
{
  extern __constant__ float GK[5][5];

  __global__ void float2uchar(
    const float *const __restrict__ in,
    unsigned char *const __restrict__ out,
    const int width,
    const int height,
    const int pitchIn,
    const int pitchOut);

  // Greyscale conversion
  __global__ void rgb2mono(
    const unsigned char *const __restrict__ rgb,
    unsigned char *const __restrict__ mono,
    const int width,
    const int height,
    const int pitchIn,
    const int pitchOut);

  // Gaussian Filter
  __global__ void gaussianFilter5x5(
    const unsigned char *const __restrict__ mono,
    unsigned char *const __restrict__ blurr,
    const int width,
    const int height,
    const int monoPitch,
    const int blurrPitch);

  // Sobel X and Y
  __global__ void sobelXY(
    const unsigned char *const __restrict__ blurr,
    float *const __restrict__ sobelX,
    float *const __restrict__ sobelY,
    const int width,
    const int height,
    const int blurrPitch,
    const int sobelXPitch,
    const int sobelYPitch);

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
    const int slopePitch);

  // Non Maximum Suppression
  __global__ void nonMaxSuppr(
    const float *const __restrict__ grad,
    const float *const __restrict__ slope,
    unsigned char *const __restrict__ nms,
    const int width,
    const int height,
    const int gradPitch,
    const int slopePitch,
    const int nmsPitch);

  // Double Threshold, finding Final Edges and Candidate Edges
  __global__ void doubleThreshold(
    const unsigned char *const __restrict__ nms,
    unsigned char *const __restrict__ thresh,
    const int width,
    const int height,
    const int nmsPitch,
    const int threshPitch,
    const unsigned char lowThreshold,
    const unsigned char highThreshold);

  __global__ void hysteresis(
    const unsigned char *const __restrict__ thresh,
    unsigned char *const __restrict__ hyster,
    int *const __restrict__ isImageModified,
    const int width,
    const int height,
    const int threshPitch,
    const int hysterPitch);

  __global__ void removeCandidates(
    const unsigned char *const __restrict__ in,
    unsigned char *const __restrict__ out,
    const int width,
    const int height,
    const int inPitch,
    const int outPitch);

}// namespace cuda
}// namespace cvp

//#endif