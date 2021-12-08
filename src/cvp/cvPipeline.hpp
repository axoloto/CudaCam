#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>

// Compiled with NVCC but not including any cuda calls to be used by Cpp Compilers further on

namespace cvp
{
// Forward declaration of cuda class
namespace cuda
{
  template<class T, size_t nbChannels>
  class CannyEdge;

  using CannyEdgeRGB8U = CannyEdge<uint8_t, 3>;
}// namespace cuda

class cvPipeline
{
public:
  cvPipeline(const unsigned int pbo, const unsigned int inputImageCols, const unsigned int inputImageRows);
  ~cvPipeline();

  bool start();
  bool stop();
  bool process(cv::Mat inputImage);

  bool isGaussianFilterEnabled() const { return m_isGaussianFilterEnabled; }
  void enableGaussianFilter(bool enable) { m_isGaussianFilterEnabled = enable; }

private:
  bool isCudaProcReady();

  bool m_isGaussianFilterEnabled;

  std::unique_ptr<cuda::CannyEdgeRGB8U> m_cudaCannyEdge;
};
}// namespace cvp
