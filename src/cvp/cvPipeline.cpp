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
  else if (inputImage.type() != CV_8UC3)
  {
    LOG_ERROR("Only supporting CV_8UC3 input type for now");
    return false;
  }

  m_cudaCannyEdge->run(inputImage, finalStage);

  //cv::Mat mat;
  //mat.create(inputImage.size(), CV_8UC1);
  //m_cudaCannyEdge->unloadImage(mat.ptr());

  return true;
}