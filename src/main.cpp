#include <functional>
#include <iostream>

#include <docopt/docopt.h>

#include "cvPipeline.hpp"
#include "logging.hpp"
#include "imguiApp.hpp"

int main(int, const char **)
{
  Utils::InitializeLogger();

#ifdef USE_IMGUI
  App::ImguiApp app;

  if (app.isInit())
  {
    app.run();
  }
#else
  cvPipeline pipeline;

  bool keepRunning = true;
  while (keepRunning)
  {
    pipeline.process();

    cv::Mat frame = pipeline.frame();

    cv::imshow("Live", frame);

    if (cv::waitKey(5) >= 0)
      keepRunning = false;
  }

#endif

  return 0;
}
