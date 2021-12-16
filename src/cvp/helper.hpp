#pragma once
#include "logging.hpp"

template<typename T>
void check(T result, char const *const func)
{
  if (result)
  {
    LOG_ERROR("CUDA errorcode={}({}) {}", static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    std::exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val)
