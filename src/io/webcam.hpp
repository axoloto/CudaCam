#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>

// Wrapper around opencv videoCapture

namespace io
{
class webcam
{
public:
  webcam();
  ~webcam();

  bool start();
  bool stop();
  bool read();

  cv::Mat frame() const { return m_frame; };

  bool isRunning() const { return m_webcam && m_webcam->isOpened(); };

private:
  cv::Mat m_frame;
  std::unique_ptr<cv::VideoCapture> m_webcam;
};
}// namespace io
