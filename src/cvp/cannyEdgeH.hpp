#pragma once

#ifdef _WIN32// Necessary for OpenGL
#include <windows.h>
#endif

#include <opencv2/core.hpp>
#include <cuda_gl_interop.h>
#include "define.hpp"

namespace cvp
{
namespace cuda
{
  // Edge detection operator using multi-stage algorithm to detect edges in images invented by John F. Canny
  // A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 8, 1986, pp. 679-698.
  class CannyEdge
  {
  public:
    CannyEdge(unsigned int pbo, unsigned int imageWidth, unsigned int imageHeight, int imageNbChannels);
    ~CannyEdge();

    void run(cv::Mat input, cvp::CannyStage finalStage);

    void setLowThreshold(unsigned char low) { m_lowThresh = min(low, m_highThresh); }
    unsigned char getLowThreshold() const { return m_lowThresh; }

    void setHighThreshold(unsigned char high) { m_highThresh = max(high, m_lowThresh); }
    unsigned char getHighThreshold() const { return m_highThresh; }

  private:
    void _initAlloc();
    void _endAlloc();

    void _loadInputImage(cv::Mat input);
    // Step 0 - Gray Conversion
    void _runGrayConversion();
    // Step 1 - Noise Reduction
    void _runGaussianFilter();
    // Step 2 - Gradient Calculation
    void _runGradient();
    // Step 3 - Non-maximum suppression
    void _runNonMaxSuppr();
    // Step 4 - Double threshold
    void _runDoubleThresh();
    // Step 5 - Edge Tracking by hysteresis
    void _runHysteresis();

    void _sendOutputToOpenGL(cvp::CannyStage finalStage);

    // Most of following int variables should be unsigned
    // sticking to signed ones to facilitate nvcc optimizations

    // Input Image Properties
    int m_inputW;
    int m_inputH;
    int m_inputNbChannels;

    int m_inputBlockSize;

    unsigned char *d_rgb;
    int d_rgbPitch;

    unsigned char *d_mono;
    int d_monoPitch;

    unsigned char *d_blurr;
    int d_blurrPitch;

    float *d_sobelX;
    int d_sobelXPitch;

    float *d_sobelY;
    int d_sobelYPitch;

    float *d_grad;
    int d_gradPitch;

    float *d_slope;
    int d_slopePitch;

    unsigned char *d_nms;
    int d_nmsPitch;

    unsigned char *d_thresh;
    int d_threshPitch;
    unsigned char m_lowThresh;
    unsigned char m_highThresh;

    unsigned char *d_hyster;
    unsigned char *d_hysterTemp;
    int d_hysterPitch;
    int d_hysterTempPitch;
    int *d_isImageModified;

    struct cudaGraphicsResource *d_pbo;

    bool m_isAlloc;
    bool m_isInit;
  };
}// namespace cuda
}// namespace cvp
