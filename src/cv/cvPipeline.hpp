#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <memory>

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

private:
  cv::Mat m_frame;
  std::unique_ptr<cv::VideoCapture> m_webcam;
};
