#include <opencv2/core.hpp>
#include <iostream>

#include "cvPipeline.hpp"
#include "cannyEdge.cuh"

void cvPipeline::test() {

  cv::Mat mat;

  if (mat.empty()) {
    std::cout << "Success!";
  }

  wrap_test_print();
}
