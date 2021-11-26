

#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdge.cuh"
#include "logging.hpp"

cvPipeline::cvPipeline()
{
  start();
}

cvPipeline::~cvPipeline()
{
  stop();
}

bool cvPipeline::start()
{
  if (m_isRunning)
  {
    LOG_ERROR("Cannot start the webcam stream, it is already running.");
    return false;
  }

  // Default camera and API
  int deviceID = 0;
  int apiID = cv::CAP_ANY;

  if (!m_webcam)
    m_webcam = std::make_unique<cv::VideoCapture>();

  m_webcam->open(0, cv::CAP_ANY);

  m_isRunning = m_webcam->isOpened();

  if (!m_isRunning)
    LOG_ERROR("Cannot open webcam stream");

  LOG_DEBUG("Webcam stream started");

  return m_isRunning;
}

bool cvPipeline::stop()
{
  if (!m_isRunning)
  {
    LOG_ERROR("Cannot stop the webcam stream, it is not running.");
    return false;
  }

  m_webcam->release();

  m_isRunning = false;

  LOG_DEBUG("Webcam stream stopped");

  return true;
}

bool cvPipeline::process()
{
  for (;;)
  {
    cv::Mat frame;

    // wait for a new frame from camera and store it into 'frame'
    m_webcam->read(frame);
    // check if we succeeded
    if (frame.empty())
    {
      std::cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // show live and wait for a key with timeout long enough to show images
    cv::imshow("Live", frame);
    if (cv::waitKey(5) >= 0)
      break;
  }

  return true;
}


/*
void cvPipeline::start()
{
  cv::Mat frame;
  cv::VideoCapture cap;

  // Default camera and API
  int deviceID = 0;
  int apiID = cv::CAP_ANY;

  cap.open(deviceID, apiID);
  if (!cap.isOpened())
  {
    std::cerr << "ERROR! Unable to open camera\n";
    return;
  }

  LOG_INFO("Hello CvPipeline");

  for (;;)
  {
    // wait for a new frame from camera and store it into 'frame'
    cap.read(frame);
    // check if we succeeded
    if (frame.empty())
    {
      std::cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // show live and wait for a key with timeout long enough to show images
    cv::imshow("Live", frame);
    if (cv::waitKey(5) >= 0)
      break;
  }

  wrap_test_print();
}
*/