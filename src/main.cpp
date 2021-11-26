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
  cvPipeline pipe;

  for (int i = 0; i > -1;)
  {
    int j = 0;
    ++j;
  }
#endif

  return 0;
}
