#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>


namespace cvp
{
namespace cuda
{
  class Convolution2D_RGB;
}

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

  bool isGLCudaInteropEnabled() const { return m_isGLCudaInteropEnabled; };

  bool isGaussianFilterEnabled() const { return m_isGaussianFilterEnabled; }
  void enableGaussianFilter(bool enable) { m_isGaussianFilterEnabled = enable; }

private:
  bool m_isGLCudaInteropEnabled;
  bool m_isGaussianFilterEnabled;

  cv::Mat m_frame;
  std::unique_ptr<cv::VideoCapture> m_webcam;
  std::unique_ptr<cuda::Convolution2D_RGB> m_cudaConv;
};
}// namespace cvp
