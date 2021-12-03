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
  cvPipeline();
  ~cvPipeline();

  bool start();
  bool stop();
  bool process();
  bool isRunning() const { return m_webcam && m_webcam->isOpened(); };
  cv::Mat frame() const { return m_frame; };

  bool isCudaProcEnabled() const { return m_isCudaProcEnabled; }
  void enableCudaProc(bool enable) { m_isCudaProcEnabled = enable; }

  bool isGLCudaInteropEnabled() const { return m_isGLCudaInteropEnabled; };

  bool isGaussianFilterEnabled() const { return m_isGaussianFilterEnabled; }
  void enableGaussianFilter(bool enable) { m_isGaussianFilterEnabled = enable; }

private:
  bool isCudaReady();

  bool m_isCudaProcEnabled;
  bool m_isGLCudaInteropEnabled;
  bool m_isGaussianFilterEnabled;

  cv::Mat m_frame;
  std::unique_ptr<cv::VideoCapture> m_webcam;
  std::unique_ptr<cuda::CannyEdgeRGB8U> m_cudaCannyEdge;
};
}// namespace cvp
