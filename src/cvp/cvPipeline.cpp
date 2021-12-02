#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdge.cuh"
#include "logging.hpp"

using namespace cvp;

cvPipeline::cvPipeline() : m_isGLCudaInteropEnabled(false)
{
  m_cudaConv = std::make_unique<cuda::Convolution2D_RGB>();
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

  // Tentative to take HD stream
  m_webcam->set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  m_webcam->set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

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
  else if (m_frame.type() != CV_8UC3)
  {
    LOG_ERROR("Only process CV_8UC3 input type");
    return false;
  }
  else
  {
    m_cudaConv->run(m_frame.ptr(), m_frame.step, m_frame.cols, m_frame.rows, 30);
    return true;
  }
}