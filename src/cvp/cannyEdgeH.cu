
#include "cannyEdgeH.hpp"
#include "cannyEdgeD.hpp"

#include "logging.hpp"
#include "define.hpp"
#include "helper.hpp"

namespace cvp
{
namespace cuda
{
  CannyEdge::CannyEdge(unsigned int pbo, unsigned int imageWidth, unsigned int imageHeight, int imageNbChannels)
    : m_isAlloc(false),
      m_inputW(imageWidth),
      m_inputH(imageHeight),
      m_inputNbChannels(imageNbChannels),
      m_inputBlockSize(32),
      m_lowThresh(10),
      m_highThresh(60)
  {
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&d_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));
    _initAlloc();
  };

  CannyEdge::~CannyEdge()
  {
    _endAlloc();
  }

  void CannyEdge::run(cv::Mat input, cvp::CannyStage finalStage)// Need to add checks
  {
    if (!m_isAlloc)
    {
      LOG_ERROR("Cannot Process Canny Edge Filter, missing allocation");
    }

    LOG_DEBUG("Start Canny Edge Filter on CUDA device");

    switch (finalStage)
    {
    case cvp::CannyStage::MONO:
    {
      _loadInputImage(input);
      _runGrayConversion();
      break;
    }
    case cvp::CannyStage::GAUSSIAN:
    {
      _loadInputImage(input);
      _runGrayConversion();
      _runGaussianFilter();
      break;
    }
    case cvp::CannyStage::GRADIENT:
    {
      _loadInputImage(input);
      _runGrayConversion();
      _runGaussianFilter();
      _runGradient();
      break;
    }
    case cvp::CannyStage::NMS:
    {
      _loadInputImage(input);
      _runGrayConversion();
      _runGaussianFilter();
      _runGradient();
      _runNonMaxSuppr();
      break;
    }
    case cvp::CannyStage::THRESH:
    {
      _loadInputImage(input);
      _runGrayConversion();
      _runGaussianFilter();
      _runGradient();
      _runNonMaxSuppr();
      _runDoubleThresh();
      break;
    }
    case cvp::CannyStage::HYSTER:
    {
      _loadInputImage(input);
      _runGrayConversion();
      _runGaussianFilter();
      _runGradient();
      _runNonMaxSuppr();
      _runDoubleThresh();
      _runHysteresis();
      break;
    }
    default:
    {
      LOG_ERROR("Canny Stage Not Recognized");
    }
    }

    _sendOutputToOpenGL(finalStage);

    LOG_DEBUG("End Canny Edge Filter on CUDA device");
  }

  void CannyEdge::_loadInputImage(cv::Mat inputImage)
  {
    if (inputImage.rows != m_inputH
        || inputImage.cols != m_inputW
        || inputImage.channels() != m_inputNbChannels)
    {
      LOG_ERROR("Cannot load image to GPU, specs different since initialization");
      return;
    }

    if (inputImage.type() == CV_8UC3)
    {
      LOG_DEBUG("Start loading RGB image in GPU");

      checkCudaErrors(cudaMemcpy2D(d_rgb, d_rgbPitch, inputImage.ptr(), inputImage.step, m_inputW * m_inputNbChannels, m_inputH, cudaMemcpyHostToDevice));

      LOG_DEBUG("End loading RGB image in GPU");
    }
    else if (inputImage.type() == CV_8UC1)
    {
      LOG_DEBUG("Start loading mono image in GPU");

      checkCudaErrors(cudaMemcpy2D(d_mono, d_monoPitch, inputImage.ptr(), inputImage.step, m_inputW * m_inputNbChannels, m_inputH, cudaMemcpyHostToDevice));

      LOG_DEBUG("End loading mono image in GPU");
    }
    else
    {
      LOG_ERROR("Only supporting CV_8UC3 input type for now");
    }
  }

  void CannyEdge::_sendOutputToOpenGL(CannyStage finalStage)
  {
    LOG_DEBUG("Start sending output image to OpenGL");

    size_t dumbSize;
    unsigned char *outPbo;
    checkCudaErrors(cudaGraphicsMapResources(1, &d_pbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&outPbo, &dumbSize, d_pbo));

    switch (finalStage)
    {
    case cvp::CannyStage::MONO:
    {
      checkCudaErrors(cudaMemcpy2D(outPbo, m_inputW, d_mono, d_monoPitch, m_inputW, m_inputH, cudaMemcpyDeviceToDevice));
      break;
    }
    case cvp::CannyStage::GAUSSIAN:
    {
      checkCudaErrors(cudaMemcpy2D(outPbo, m_inputW, d_blurr, d_blurrPitch, m_inputW, m_inputH, cudaMemcpyDeviceToDevice));
      break;
    }
    case cvp::CannyStage::GRADIENT:
    {
      dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
      dim3 grid((m_inputW + m_inputBlockSize - 1) / m_inputBlockSize, (m_inputH + m_inputBlockSize - 1) / m_inputBlockSize, 1);
      float2uchar<<<grid, blocks>>>(d_grad, outPbo, m_inputW, m_inputH, d_gradPitch / sizeof(float), m_inputW);
      break;
    }
    case cvp::CannyStage::NMS:
    {
      checkCudaErrors(cudaMemcpy2D(outPbo, m_inputW, d_nms, d_nmsPitch, m_inputW, m_inputH, cudaMemcpyDeviceToDevice));
      break;
    }
    case cvp::CannyStage::THRESH:
    {
      checkCudaErrors(cudaMemcpy2D(outPbo, m_inputW, d_thresh, d_threshPitch, m_inputW, m_inputH, cudaMemcpyDeviceToDevice));
      break;
    }
    case cvp::CannyStage::HYSTER:
    {
      checkCudaErrors(cudaMemcpy2D(outPbo, m_inputW, d_hyster, d_hysterPitch, m_inputW, m_inputH, cudaMemcpyDeviceToDevice));
      break;
    }
    default:
    {
      LOG_ERROR("Canny Stage Not Recognized");
    }
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &d_pbo, 0));

    LOG_DEBUG("End sending output image to OpenGL");
  }

  void CannyEdge::_runGrayConversion()
  {
    LOG_DEBUG("Start Gray Conversion");

    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_inputW + m_inputBlockSize - 1) / m_inputBlockSize, (m_inputH + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    rgb2mono<<<grid, blocks>>>(d_rgb, d_mono, m_inputW, m_inputH, d_rgbPitch, d_monoPitch);

    LOG_DEBUG("End Gray Conversion");
  }

  void CannyEdge::_runGaussianFilter()
  {
    LOG_DEBUG("Start Gaussian Filter");

    int outputBlockSize = m_inputBlockSize - 4;//2 halo cells on each border
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_inputW + outputBlockSize - 1) / outputBlockSize, (m_inputH + outputBlockSize - 1) / outputBlockSize, 1);
    gaussianFilter5x5<<<grid, blocks>>>(d_mono, d_blurr, m_inputW, m_inputH, d_monoPitch, d_blurrPitch);

    LOG_DEBUG("End Gaussian Filter");
  }

  void CannyEdge::_runGradient()
  {
    LOG_DEBUG("Start Gradient Computation");
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);

    LOG_DEBUG(" 1-Sobel");
    int outputBlockSize = m_inputBlockSize - 2;//1 halo cell on each border
    dim3 sobelGrid((m_inputW + outputBlockSize - 1) / outputBlockSize, (m_inputH + outputBlockSize - 1) / outputBlockSize, 1);
    sobelXY<<<sobelGrid, blocks>>>(d_blurr, d_sobelX, d_sobelY, m_inputW, m_inputH, d_blurrPitch, d_sobelXPitch / sizeof(float), d_sobelYPitch / sizeof(float));

    LOG_DEBUG(" 2-Gradient");
    dim3 gradGrid((m_inputW + m_inputBlockSize - 1) / m_inputBlockSize, (m_inputH + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    gradSlope<<<gradGrid, blocks>>>(d_sobelX, d_sobelY, d_grad, d_slope, m_inputW, m_inputH, d_sobelXPitch / sizeof(float), d_sobelYPitch / sizeof(float), d_gradPitch / sizeof(float), d_slopePitch / sizeof(float));

    LOG_DEBUG("End Gradient Computation");
  }

  void CannyEdge::_runNonMaxSuppr()
  {
    LOG_DEBUG("Start Non Max Suppression");

    int outputBlockSize = m_inputBlockSize - 2;
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_inputW + outputBlockSize - 1) / outputBlockSize, (m_inputH + outputBlockSize - 1) / outputBlockSize, 1);
    nonMaxSuppr<<<grid, blocks>>>(d_grad, d_slope, d_nms, m_inputW, m_inputH, d_gradPitch / sizeof(float), d_slopePitch / sizeof(float), d_nmsPitch);

    LOG_DEBUG("End Non Max Suppression");
  }

  void CannyEdge::_runDoubleThresh()
  {
    LOG_DEBUG("Start Double Threshold");

    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_inputW + m_inputBlockSize - 1) / m_inputBlockSize, (m_inputH + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    doubleThreshold<<<grid, blocks>>>(d_nms, d_thresh, m_inputW, m_inputH, d_nmsPitch, d_threshPitch, m_lowThresh, m_highThresh);

    LOG_DEBUG("End Double Threshold");
  }

  void CannyEdge::_runHysteresis()
  {
    LOG_DEBUG("Start Hysteresis");

    int outputBlockSize = m_inputBlockSize - 2;
    dim3 blocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 grid((m_inputW + outputBlockSize - 1) / outputBlockSize, (m_inputH + outputBlockSize - 1) / outputBlockSize, 1);

    int isImageModified = 0;
    cudaMemcpy(d_isImageModified, &isImageModified, sizeof(int), cudaMemcpyHostToDevice);
    hysteresis<<<grid, blocks>>>(d_thresh, d_hyster, d_isImageModified, m_inputW, m_inputH, d_threshPitch, d_hysterPitch);
    cudaMemcpy(&isImageModified, d_isImageModified, sizeof(int), cudaMemcpyDeviceToHost);

    int nbIters = 0;
    int maxNbIters = 100;//safety belt
    while (nbIters < maxNbIters && isImageModified)
    {
      std::swap(d_hyster, d_hysterTemp);
      std::swap(d_hysterPitch, d_hysterTempPitch);

      isImageModified = 0;
      cudaMemcpy(d_isImageModified, &isImageModified, sizeof(int), cudaMemcpyHostToDevice);
      hysteresis<<<grid, blocks>>>(d_hysterTemp, d_hyster, d_isImageModified, m_inputW, m_inputH, d_hysterTempPitch, d_hysterPitch);
      cudaMemcpy(&isImageModified, d_isImageModified, sizeof(int), cudaMemcpyDeviceToHost);
      nbIters++;
    }

    LOG_INFO("Number of hysteresis iterations {}, number of blocks with unfinished work {}", nbIters, isImageModified);

    std::swap(d_hyster, d_hysterTemp);
    std::swap(d_hysterPitch, d_hysterTempPitch);

    dim3 rblocks(m_inputBlockSize, m_inputBlockSize, 1);
    dim3 rgrid((m_inputW + m_inputBlockSize - 1) / m_inputBlockSize, (m_inputH + m_inputBlockSize - 1) / m_inputBlockSize, 1);
    removeCandidates<<<rgrid, rblocks>>>(d_hysterTemp, d_hyster, m_inputW, m_inputH, d_hysterTempPitch, d_hysterPitch);

    LOG_DEBUG("End Hysteresis");
  }

  void CannyEdge::_initAlloc()
  {
    LOG_DEBUG("Start allocating image memory in GPU");

    // Don't want size_t in device kernels for performance reasons
    size_t pitch = 0;
    checkCudaErrors(cudaMallocPitch(&d_rgb, &pitch, m_inputW * m_inputNbChannels, m_inputH));
    d_rgbPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_mono, &pitch, m_inputW, m_inputH));
    d_monoPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_blurr, &pitch, m_inputW, m_inputH));
    d_blurrPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_sobelX, &pitch, m_inputW * sizeof(float), m_inputH));
    d_sobelXPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_sobelY, &pitch, m_inputW * sizeof(float), m_inputH));
    d_sobelYPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_grad, &pitch, m_inputW * sizeof(float), m_inputH));
    d_gradPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_slope, &pitch, m_inputW * sizeof(float), m_inputH));
    d_slopePitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_nms, &pitch, m_inputW, m_inputH));
    d_nmsPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_thresh, &pitch, m_inputW, m_inputH));
    d_threshPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_hyster, &pitch, m_inputW, m_inputH));
    d_hysterPitch = (int)pitch;
    checkCudaErrors(cudaMallocPitch(&d_hysterTemp, &pitch, m_inputW, m_inputH));
    d_hysterTempPitch = (int)pitch;

    checkCudaErrors(cudaMalloc(&d_isImageModified, sizeof(int)));

    // Loading Gaussian Kernel 5x5 onto GPU
    std::array<std::array<float, 5>, 5> GK_CPU = { { { 2, 4, 5, 4, 2 }, { 4, 9, 12, 9, 4 }, { 5, 12, 15, 12, 5 }, { 4, 9, 12, 9, 4 }, { 2, 4, 5, 4, 2 } } };
    for (int i = 0; i < 5; ++i)
    {
      for (int j = 0; j < 5; ++j)
      {
        GK_CPU[i][j] *= 1 / 159.0f;
      }
    }
    checkCudaErrors(cudaMemcpyToSymbol(GK, GK_CPU.data(), 25 * sizeof(float)));

    LOG_DEBUG("End allocating image memory in GPU");

    m_isAlloc = true;
  }

  void CannyEdge::_endAlloc()
  {
    LOG_DEBUG("Start deallocating image memory in GPU");

    checkCudaErrors(cudaFree(d_rgb));
    checkCudaErrors(cudaFree(d_mono));
    checkCudaErrors(cudaFree(d_blurr));
    checkCudaErrors(cudaFree(d_sobelX));
    checkCudaErrors(cudaFree(d_sobelY));
    checkCudaErrors(cudaFree(d_grad));
    checkCudaErrors(cudaFree(d_slope));
    checkCudaErrors(cudaFree(d_nms));
    checkCudaErrors(cudaFree(d_thresh));
    checkCudaErrors(cudaFree(d_hyster));
    checkCudaErrors(cudaFree(d_hysterTemp));
    checkCudaErrors(cudaFree(d_isImageModified));

    LOG_DEBUG("End deallocating image memory in GPU");

    m_isAlloc = false;
  }
}// namespace cuda
}// namespace cvp