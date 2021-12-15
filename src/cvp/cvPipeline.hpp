#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "define.hpp"

#include <memory>

// Proxy compiled with NVCC but not including any cuda calls to be used by Cpp Compilers further on

namespace cvp
{
// Forward declaration of cuda class
namespace cuda
{
  template<class T, size_t nbChannels>
  class CannyEdge;

  using CannyEdgeRGB8U = CannyEdge<unsigned char, 3>;
}// namespace cuda

class cvPipeline
{
public:
  cvPipeline(const unsigned int pbo, const unsigned int inputImageCols, const unsigned int inputImageRows);
  ~cvPipeline();

  bool process(cv::Mat inputImage, CannyStage finalStage);

  void setLowThreshold(unsigned char low);
  unsigned char getLowThreshold();

  void setHighThreshold(unsigned char high);
  unsigned char getHighThreshold();

private:
  std::unique_ptr<cuda::CannyEdgeRGB8U> m_cudaCannyEdge;
};
}// namespace cvp
