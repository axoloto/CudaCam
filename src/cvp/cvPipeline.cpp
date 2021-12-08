#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdge.cuh"
#include "logging.hpp"

using namespace cvp;

cvPipeline::cvPipeline(unsigned int pbo) : m_isCudaProcEnabled(true)
{
  if (start())
  {
    // Reading webcam stream once to initialize cuda buffers
    m_webcam->read(m_inputFrame);
    if (!m_inputFrame.empty())
    {
      m_cudaCannyEdge = std::make_unique<cuda::CannyEdgeRGB8U>(pbo, m_inputFrame.cols, m_inputFrame.rows);
    }
  }
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

bool cvPipeline::isCudaProcReady()
{
  return (m_cudaCannyEdge.get() != nullptr);
}

bool cvPipeline::process()
{
  if (!isRunning())
  {
    LOG_ERROR("Cannot process the webcam stream, it is not running.");
    return false;
  }

  // wait for a new frame from camera and store it into m_inputFrame
  m_webcam->read(m_inputFrame);

  if (m_inputFrame.empty())
  {
    LOG_ERROR("Blank frame grabbed");
    return false;
  }
  else if (m_inputFrame.type() != CV_8UC3)
  {
    LOG_ERROR("Only process CV_8UC3 input type");
    return false;
  }

  if (m_frameOut.empty())
  {
    m_frameOut.create(m_inputFrame.rows, m_inputFrame.cols, CV_8UC1);
  }

  if (!m_isCudaProcEnabled || !isCudaProcReady())
    return false;

  m_cudaCannyEdge->loadInputImage(m_inputFrame.ptr(), m_inputFrame.step);

  m_cudaCannyEdge->runGrayConversion(30);// UI informed with blocked option

  // if (m_isGaussianFilterEnabled)
  //   m_cudaCannyEdge->runGaussianFilter(30);

  // m_cudaCannyEdge->unloadImage(m_inputFrame.ptr());
  //m_cudaCannyEdge->unloadImage(m_frameOut.ptr());

  return true;
}