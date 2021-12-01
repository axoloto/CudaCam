#pragma once

#include <stdio.h>

void wrap_test_print();

namespace cvp
{
    namespace cuda
    {
        void convolution2D(unsigned char* imageCPU, unsigned int pitchCPU, unsigned int imageWidth, unsigned int imageHeight, unsigned int blockSize);
    }
} // namespace cvp
