#pragma once

#include <SDL.h>
#include <imgui.h>
#include <string>
#include <array>

namespace App {
class ImguiApp {
public:
  ImguiApp();
  ~ImguiApp() = default;
  void run();
  bool isInit() const { return m_init; }

private:
  bool initWindow();
  bool closeWindow();
  bool checkSDLStatus();

  SDL_Window *m_window;
  SDL_GLContext m_OGLContext;

  std::string m_nameApp;

  // FPS (Frame per second or framerate)
  // User-defined target framerate
  int m_targetFps;
  // Real framerate
  // can be lower than target depending on the physics simulation cost
  float m_currFps;

  std::array<int, 2> m_windowSize;
  std::array<float, 4> m_backGroundColor;
  bool m_init;
};

}// namespace App