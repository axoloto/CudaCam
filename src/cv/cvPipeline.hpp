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

private:
  bool m_isRunning;

  std::unique_ptr<cv::VideoCapture> m_webcam;
};
