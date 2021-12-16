#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdgeH.hpp"
#include "logging.hpp"

using namespace cvp;

cvPipeline::cvPipeline(const unsigned int pbo, const unsigned int inputImageCols, const unsigned int inputImageRows, const int inputNbChannels)
{
  m_cudaCannyEdge = std::make_unique<cuda::CannyEdge>(pbo, inputImageCols, inputImageRows, inputNbChannels);
}

cvPipeline::~cvPipeline()
{
  m_cudaCannyEdge.release();
}

bool cvPipeline::process(cv::Mat inputImage, CannyStage finalStage)
{
  if (!m_cudaCannyEdge.get())
  {
    LOG_ERROR("Cannot process the webcam stream, Cuda is not ready.");
    return false;
  }

  if (inputImage.empty())
  {
    LOG_ERROR("Blank frame grabbed");
    return false;
  }
  else if (inputImage.type() != CV_8UC3 && inputImage.type() != CV_8UC1)
  {
    LOG_ERROR("Only supporting CV_8UC3 and CV_8UC1 input types for now");
    return false;
  }

  m_cudaCannyEdge->run(inputImage, finalStage);

  return true;
}

void cvPipeline::setLowThreshold(unsigned char low)
{
  if (!m_cudaCannyEdge.get())
  {
    LOG_ERROR("Cannot modify low threshold, Cuda is not ready.");
    return;
  }
  m_cudaCannyEdge->setLowThreshold(low);
}

unsigned char cvPipeline::getLowThreshold()
{
  if (!m_cudaCannyEdge.get())
  {
    LOG_ERROR("Cannot modify low threshold, Cuda is not ready.");
    return 0;
  }
  return m_cudaCannyEdge->getLowThreshold();
}

void cvPipeline::setHighThreshold(unsigned char high)
{
  if (!m_cudaCannyEdge.get())
  {
    LOG_ERROR("Cannot modify high threshold, Cuda is not ready.");
    return;
  }
  m_cudaCannyEdge->setHighThreshold(high);
}

unsigned char cvPipeline::getHighThreshold()
{
  if (!m_cudaCannyEdge.get())
  {
    LOG_ERROR("Cannot modify low threshold, Cuda is not ready.");
    return 255;
  }
  return m_cudaCannyEdge->getHighThreshold();
}
