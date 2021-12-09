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

bool ImguiApp::initWebcam()
{
  m_webcam = std::make_unique<io::webcam>();

  return (m_webcam && !m_webcam->frame().empty());
}

bool ImguiApp::initOpenGL()
{
  if (!gladLoadGL())
    return false;

  // Pixel buffer object filled by Cuda
  m_pboCols = m_webcam->frame().cols;
  m_pboRows = m_webcam->frame().rows;
  glGenBuffers(1, &m_pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, m_pboCols * m_pboRows * sizeof(unsigned char), nullptr, GL_STREAM_DRAW_ARB);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  glGenTextures(1, &m_texture);
  glBindTexture(GL_TEXTURE_2D, m_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glBindTexture(GL_TEXTURE_2D, 0);

  return true;
}

bool ImguiApp::initCvPipeline()
{
  m_cvPipeline = std::make_unique<cvp::cvPipeline>(m_pbo, m_webcam->frame().cols, m_webcam->frame().rows);

  return m_cvPipeline.get();
}

bool ImguiApp::closeWindow()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(m_OGLContext);
  SDL_DestroyWindow(m_window);
  SDL_Quit();

  glDeleteBuffers(1, &m_pbo);
  glDeleteTextures(1, &m_texture);
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

ImguiApp::ImguiApp()
  : m_nameApp("CudaCam " + Utils::GetVersions()),
    m_backGroundColor({ 0.0f, 0.0f, 0.0f, 1.00f }),
    m_windowSize({ 1920, 1000 }),//1080
    m_targetFps(60),
    m_texture(0),
    m_pboCols(0),
    m_pboRows(0),
    m_totalTimeMs(0),
    m_totalFrames(0),
    m_isCvPipelineEnabled(true),
    m_now(std::chrono::steady_clock::now()),
    m_init(false)
{
  if (!initWindow())
  {
    LOG_ERROR("Failed to initialize application window");
    return;
  }

  if (!initWebcam())
  {
    LOG_ERROR("Failed to initialize webcam");
    return;
  }

  if (!initOpenGL())
  {
    LOG_ERROR("Failed to initialize openGL loader");
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

  m_totalTimeMs += std::chrono::duration_cast<std::chrono::milliseconds>(timeSpent).count();
  m_totalFrames++;

  float avgTimeMs = m_totalTimeMs / m_totalFrames;
  float fps = 1000.0f / avgTimeMs;

  if (m_totalTimeMs > 10000)
  {
    m_totalTimeMs = 0;
    m_totalFrames = 0;
  }

  ImGui::Text(" %.3f ms/frame (%.1f FPS) ", avgTimeMs, fps);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (m_cvPipeline)
  {
    ImGui::Checkbox("Cuda Processing", &m_isCvPipelineEnabled);

    if (m_isCvPipelineEnabled)
    {
      bool isGaussianFilterEnabled = m_cvPipeline->isGaussianFilterEnabled();
      if (ImGui::Checkbox("Noise Reduction", &isGaussianFilterEnabled))
      {
        m_cvPipeline->enableGaussianFilter(isGaussianFilterEnabled);
      }
    }
  }

  ImGui::End();
}

void ImguiApp::displayLiveStream()
{
  if (!m_isCvPipelineEnabled)
  {
    cv::Mat image = m_webcam->frame();

    if (!image.empty() && image.type() == CV_8UC3)
    {
      //cv::flip(image, image, 1);
      cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

      glBindTexture(GL_TEXTURE_2D, m_texture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.ptr());
      glBindTexture(GL_TEXTURE_2D, 0);

      ImGui::SetNextWindowPos(ImVec2(150, 120), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(image.cols, image.rows), ImGuiCond_FirstUseEver);
      ImGui::Begin("Live Stream");
      ImGui::Text("%d x %d", image.cols, image.rows);
      ImGui::Image((void *)(intptr_t)m_texture, ImVec2(image.cols, image.rows));
      ImGui::End();
    }
    else
    {
      LOG_ERROR("Cannot display webcam image");
    }
  }
  else
  {
    // Interop OpenGL/CUDA
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_pboCols, m_pboRows, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    ImGui::SetNextWindowPos(ImVec2(150, 120), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(m_pboCols, m_pboRows), ImGuiCond_FirstUseEver);
    ImGui::Begin("Cuda Live Stream");
    ImGui::Text("%d x %d", m_pboCols, m_pboRows);
    ImGui::Image((void *)(intptr_t)m_texture, ImVec2(m_pboCols, m_pboRows));
    ImGui::End();
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

    m_webcam->read();

    m_cvPipeline->process(m_webcam->frame());

    displayLiveStream();

    runLoopEnder();
  }
}