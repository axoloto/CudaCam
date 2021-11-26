

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
  if (isRunning())
  {
    LOG_ERROR("Cannot start the webcam stream, it is already running.");
    return false;
  }

  if (!m_webcam)
    m_webcam = std::make_unique<cv::VideoCapture>();

  if (!m_webcam)
  {
    LOG_ERROR("Cannot start the webcam stream");
    return false;
  }

  m_webcam->open(0, cv::CAP_ANY);

  if (!isRunning())
    LOG_ERROR("Cannot open webcam stream");
  else
    LOG_DEBUG("Webcam stream started");

  return isRunning();
}

bool cvPipeline::stop()
{
  if (!isRunning())
  {
    LOG_ERROR("Cannot stop the webcam stream, it is not running.");
    return false;
  }

  m_webcam->release();

  LOG_DEBUG("Webcam stream stopped");

  return true;
}

bool cvPipeline::process()
{
  if (!isRunning())
  {
    LOG_ERROR("Cannot process the webcam stream, it is not running.");
    return false;
  }

  // wait for a new frame from camera and store it into m_frame
  m_webcam->read(m_frame);

  if (m_frame.empty())
  {
    LOG_ERROR("Blank frame grabbed");
    return false;
  }
  else
    return true;
}