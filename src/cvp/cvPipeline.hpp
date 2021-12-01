#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>

namespace cvp
{
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

private:
  bool m_isGLCudaInteropEnabled;

  cv::Mat m_frame;
  std::unique_ptr<cv::VideoCapture> m_webcam;
};
}// namespace cvp
