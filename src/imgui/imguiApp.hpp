#pragma once

#include <glad/glad.h>
#include <SDL2/SDL.h>
#undef main//Thank you SDL

#include <string>
#include <array>
#include <memory>
#include <chrono>

#include "cvPipeline.hpp"

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
  bool initCvPipeline();
  bool initOpenGL();

  bool closeWindow();
  bool checkSDLStatus();

  void displayMainWidget();
  void displayLiveStream();

  void runLoopStarter();
  void runLoopEnder();

  SDL_Window *m_window;
  SDL_GLContext m_OGLContext;

  std::string m_nameApp;

  // FPS (Frame per second or framerate)
  // User-defined target framerate
  int m_targetFps;
  // Real framerate
  // can be lower than target depending on the physics simulation cost
  float m_currFps;

  // Rendering
  GLuint m_pbo;
  GLuint m_texture;

  std::array<int, 2> m_windowSize;
  std::array<float, 4> m_backGroundColor;
  bool m_init;

  std::unique_ptr<cvp::cvPipeline> m_cvPipeline;

  std::chrono::steady_clock::time_point m_now;
};

}// namespace App