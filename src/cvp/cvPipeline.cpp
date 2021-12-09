#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdge.cuh"
#include "logging.hpp"

using namespace cvp;

cvPipeline::cvPipeline(const unsigned int pbo, const unsigned int inputImageCols, const unsigned int inputImageRows)
{
  m_cudaCannyEdge = std::make_unique<cuda::CannyEdgeRGB8U>(pbo, inputImageCols, inputImageRows);
}

cvPipeline::~cvPipeline()
{
  int i = 0;
}

bool cvPipeline::isCudaProcReady()
{
  return (m_cudaCannyEdge.get() != nullptr);
}

bool cvPipeline::process(cv::Mat inputImage)
{
  if (!isCudaProcReady())
  {
    LOG_ERROR("Cannot process the webcam stream, Cuda is not ready.");
    return false;
  }

  if (inputImage.empty())
  {
    LOG_ERROR("Blank frame grabbed");
    return false;
  }
  else if (inputImage.type() != CV_8UC3)
  {
    LOG_ERROR("Only supporting CV_8UC3 input type for now");
    return false;
  }

  m_cudaCannyEdge->loadInputImage(inputImage.ptr(), inputImage.step);

  m_cudaCannyEdge->runGrayConversion(30);// UI informed with blocked option

  if (m_isGaussianFilterEnabled)
    m_cudaCannyEdge->runGaussianFilter();

  // m_cudaCannyEdge->unloadImage(m_inputFrame.ptr());
  //m_cudaCannyEdge->unloadImage(m_frameOut.ptr());

  return true;
}