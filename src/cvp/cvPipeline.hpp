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
  cvPipeline(unsigned int pbo);
  ~cvPipeline();

  bool start();
  bool stop();
  bool process();

  cv::Mat inputFrame() const { return m_inputFrame; };

  bool isRunning() const { return m_webcam && m_webcam->isOpened(); };

  bool isCudaProcEnabled() const { return m_isCudaProcEnabled; }
  void enableCudaProc(bool enable) { m_isCudaProcEnabled = enable; }

  bool isGaussianFilterEnabled() const { return m_isGaussianFilterEnabled; }
  void enableGaussianFilter(bool enable) { m_isGaussianFilterEnabled = enable; }

private:
  bool isCudaProcReady();

  bool m_isCudaProcEnabled;
  bool m_isGaussianFilterEnabled;

  cv::Mat m_inputFrame;
  cv::Mat m_frameOut;//WIP
  std::unique_ptr<cv::VideoCapture> m_webcam;
  std::unique_ptr<cuda::CannyEdgeRGB8U> m_cudaCannyEdge;
};
}// namespace cvp
