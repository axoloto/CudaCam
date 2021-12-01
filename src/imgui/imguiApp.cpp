#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <glad/glad.h>

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

  // Initialize OpenGL loader
  if (!gladLoadGL())
  {
    LOG_ERROR("Failed to initialize OpenGL loader!");
    return false;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui::StyleColorsDark();

  ImGui_ImplSDL2_InitForOpenGL(m_window, m_OGLContext);
  ImGui_ImplOpenGL3_Init(GLSL_VERSION);

  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glGenTextures(1, &m_imageTexture);

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
  m_cvPipeline = std::make_unique<cvp::cvPipeline>();

  return m_cvPipeline.get();
}

ImguiApp::ImguiApp()
  : m_nameApp("CudaCam " + Utils::GetVersions()),
    m_backGroundColor({ 0.0f, 0.0f, 0.0f, 1.00f }),
    m_windowSize({ 1280, 720 }),
    m_targetFps(60),
    m_currFps(60.0f),
    m_imageTexture(0),
    m_now(std::chrono::steady_clock::now()),
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

ImguiApp::~ImguiApp()
{
  closeWindow();
}

void ImguiApp::displayMainWidget()
{
  ImGui::SetNextWindowPos(ImVec2(15, 12), ImGuiCond_FirstUseEver);
  ImGui::Begin("Main Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  if (!m_init) return;

  auto now = std::chrono::steady_clock::now();
  auto timeSpent = now - m_now;
  m_now = now;

  float timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeSpent).count();
  float fps = 1000.0f / timeMs;

  ImGui::Text(" %.3f ms/frame (%.1f FPS) ", timeMs, fps);

  ImGui::End();
}

void ImguiApp::displayLiveStream()
{
  if (!m_cvPipeline->isGLCudaInteropEnabled())
  {
    cv::Mat image = m_cvPipeline->frame();

    if (!image.empty())
    {
      glBindTexture(GL_TEXTURE_2D, m_imageTexture);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      // Set texture clamping method
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

      cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

      glTexImage2D(
        GL_TEXTURE_2D,// Type of texture
        0,// Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB,// Internal colour format to convert to
        image.cols,// Image width
        image.rows,// Image height
        0,// Border width in pixels (can either be 1 or 0)
        GL_RGB,// Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,// Image data type
        image.ptr());// The actual image data itself

      ImGui::Begin("Live Stream");
      ImGui::Text("%d x %d", image.cols, image.rows);
      ImGui::Image((void *)(intptr_t)m_imageTexture, ImVec2(image.cols, image.rows));
      ImGui::End();
    }
  }
}

void ImguiApp::runLoopStarter()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplSDL2_NewFrame(m_window);
  ImGui::NewFrame();

  ImGuiIO &io = ImGui::GetIO();
  glViewport(0, 0, (int)io.DisplaySize[0], (int)io.DisplaySize[1]);
  glClearColor(m_backGroundColor[0], m_backGroundColor[1], m_backGroundColor[2], m_backGroundColor[3]);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void ImguiApp::runLoopEnder()
{
  ImGui::Render();

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  SDL_GL_SwapWindow(m_window);
}

void ImguiApp::run()
{
  if (!m_init) return;

  while (checkSDLStatus())
  {
    runLoopStarter();

    displayMainWidget();

    m_cvPipeline->process();

    displayLiveStream();

    runLoopEnder();
  }
}