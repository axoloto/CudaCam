#include <iostream>

#include "webcam.hpp"
#include "logging.hpp"

using namespace io;

webcam::webcam()
{
  if (start())
    m_webcam->read(m_frame);// init frame specs
}

webcam::~webcam()
{
  stop();
}

bool webcam::start()
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

  // Tentative to take HD stream
  m_webcam->set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  m_webcam->set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

  if (!isRunning())
    LOG_ERROR("Cannot open webcam stream");
  else
    LOG_DEBUG("Webcam stream started");

  return isRunning();
}

bool webcam::stop()
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

bool webcam::read()
{
  if (!isRunning())
  {
    LOG_ERROR("Cannot read the webcam stream, it is not running.");
    return false;
  }

  // wait for a new frame from camera and store it into m_frame
  m_webcam->read(m_frame);

  if (m_frame.empty())
  {
    LOG_ERROR("Blank frame grabbed");
    return false;
  }

  return true;
}