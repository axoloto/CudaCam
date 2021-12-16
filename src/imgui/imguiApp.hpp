#pragma once

#include <glad/glad.h>
#include <SDL2/SDL.h>
#undef main//Thank you SDL

#include <imgui.h>

#include <string>
#include <array>
#include <memory>
#include <chrono>

#include "cvPipeline.hpp"
#include "webcam.hpp"

namespace App
{
class ImguiApp
{
public:
  ImguiApp();
  ~ImguiApp();
  void run();
  bool isInit() const { return m_init; }

private:
  bool initWindow();
  bool initWebcam();
  bool initOpenGL();
  bool initCvPipeline();

  bool closeWindow();
  bool checkSDLStatus();

  void displayMainWidget();
  void displayLiveStream();

  void runLoopStarter();
  void runLoopEnder();

  void zoomToolTip(ImVec2 cursorPos, ImVec2 ioPos);

  std::string m_nameApp;

  // App
  SDL_Window *m_window;
  SDL_GLContext m_OGLContext;
  std::array<int, 2> m_windowSize;
  std::array<float, 4> m_backGroundColor;
  bool m_init;

  // Fps
  std::chrono::steady_clock::time_point m_now;
  size_t m_totalTimeMs;
  size_t m_totalFrames;
  int m_targetFps;

  // UI option
  bool m_isZoomEnabled;

  // Rendering
  size_t m_pboCols, m_pboRows;
  GLuint m_pbo;
  GLuint m_texture;
  size_t m_zoom;

  // IO
  // Take last frame from the stream
  // Cleaner way would be to pause the stream
  // But opencv stream api has no break mode apparently
  bool m_takeLastFrame;
  // Video stream from webcam found by OpenCV
  std::unique_ptr<io::webcam> m_webcam;

  // Computer Vision Pipeline accelerated with CUDA
  bool m_isCvPipelineEnabled;
  std::pair<cvp::CannyStage, std::string> m_cvFinalStage;
  std::unique_ptr<cvp::cvPipeline> m_cvPipeline;
};

}// namespace App