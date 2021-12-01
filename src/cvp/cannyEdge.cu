#ifndef _CANNY_EDGE_KERNEL_
#define _CANNY_EDGE_KERNEL_

#include "cannyEdge.cuh"

__global__ void test_print(){
  printf("Hello World!\n");
}

void wrap_test_print() {
  test_print<<<1, 1>>>();
  return;
}

__global__ void conv2D(){
  int i = 0;
}

void cvp::cuda::convolution2D()
{
  conv2D<<<1,1>>>();
  return;
}

#endif