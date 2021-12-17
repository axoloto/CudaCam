#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "define.hpp"

#include <memory>

// Pure cpp proxy not including any cuda calls

namespace cvp
{
// Forward declaration of cuda class
namespace cuda
{
  class CannyEdge;
}// namespace cuda

class cvPipeline
{
public:
  cvPipeline(const unsigned int pbo, const unsigned int inputImageCols, const unsigned int inputImageRows, const int inputImageNbChannels);
  ~cvPipeline();

  bool process(cv::Mat inputImage, CannyStage finalStage);

  void setLowThreshold(unsigned char low);
  unsigned char getLowThreshold() const;

  void setHighThreshold(unsigned char high);
  unsigned char getHighThreshold() const;

  void enableCudaProfiling(bool profiling);
  bool isCudaProfilingEnabled() const;

private:
  std::unique_ptr<cuda::CannyEdge> m_cudaCannyEdge;
};
}// namespace cvp
