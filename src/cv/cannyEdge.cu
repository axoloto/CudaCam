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

#endif