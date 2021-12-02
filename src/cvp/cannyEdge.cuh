#pragma once

#include <stdio.h>
#include <stdint.h>

namespace cvp
{
namespace cuda
{
  class Convolution2D_RGB
  {
  public:
    Convolution2D_RGB();
    ~Convolution2D_RGB();
    void run(unsigned char *imageCPU, unsigned int pitchCPU, unsigned int imageWidth, unsigned int imageHeight, unsigned int blockSize);
    void initAlloc(unsigned int imageWidth, unsigned int imageHeight);
    void endAlloc();

  private:
    uint8_t *m_inImageGPU;
    uint8_t *m_outImageGPU;
    size_t m_pitchInGPU;
    size_t m_pitchOutGPU;
    bool m_isAlloc;
  };
}// namespace cuda
}// namespace cvp
