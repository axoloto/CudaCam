#include <imgui.h>
#include <bindings/imgui_impl_opengl3.h>
#include <bindings/imgui_impl_sdl.h>

#include "imguiApp.hpp"
#include "logging.hpp"
#include "utils.hpp"

using namespace App;

constexpr auto GLSL_VERSION = "#version 130";

bool ImguiApp::initWindow()
{
  // Setup SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
  {
    LOG_ERROR("Error: {}", SDL_GetError());
    return false;
  }

  // GL 3.0 + GLSL 130
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  // Create window with graphics context
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_SHOWN);
  m_window = SDL_CreateWindow(m_nameApp.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_windowSize[0], m_windowSize[1], window_flags);
  m_OGLContext = SDL_GL_CreateContext(m_window);
  SDL_GL_MakeCurrent(m_window, m_OGLContext);
  SDL_GL_SetSwapInterval(1);// Enable vsync

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui::StyleColorsDark();

  ImGui_ImplSDL2_InitForOpenGL(m_window, m_OGLContext);
  ImGui_ImplOpenGL3_Init(GLSL_VERSION);

  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  return true;
}

bool ImguiApp::closeWindow()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(m_OGLContext);
  SDL_DestroyWindow(m_window);
  SDL_Quit();

  return true;
}

bool ImguiApp::checkSDLStatus()
{
  bool keepGoing = true;

  SDL_Event event;
  while (SDL_PollEvent(&event))
  {
    ImGui_ImplSDL2_ProcessEvent(&event);
    switch (event.type)
    {
    case SDL_QUIT:
    {
      keepGoing = false;
      break;
    }
    case SDL_WINDOWEVENT:
    {
      if (event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
      {
        keepGoing = false;
      }
      if (event.window.event == SDL_WINDOWEVENT_RESIZED)
      {
        m_windowSize[0] = event.window.data1;
        m_windowSize[1] = event.window.data2;
      }
      break;
    }
    }
  }
  return keepGoing;
}

bool ImguiApp::initCvPipeline()
{
  m_cvPipeline = std::make_unique<cvPipeline>();

  return m_cvPipeline.get();
}

ImguiApp::ImguiApp()
  : m_nameApp("CudaCam " + Utils::GetVersions()),
    m_backGroundColor({ 0.0f, 0.0f, 0.0f, 1.00f }),
    m_windowSize({ 1280, 720 }),
    m_targetFps(60),
    m_currFps(60.0f),
    m_init(false)
{
  if (!initWindow())
  {
    LOG_ERROR("Failed to initialize application window");
    return;
  }

  if (!initCvPipeline())
  {
    LOG_ERROR("Failed to initialize computer vision pipeline");
    return;
  }

  LOG_INFO("Application correctly initialized");

  m_init = true;
}

void ImguiApp::run()
{
  while (checkSDLStatus())
  {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(m_window);
    ImGui::NewFrame();

    m_cvPipeline->process();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(m_window);
  }

  closeWindow();
}