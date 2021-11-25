option(USE_IMGUI "Enable compilation of ImGui sample" ON)

include_directories(cv)
add_subdirectory(cv)

# imgui app
if(USE_IMGUI)
  message("Using Dear ImGui")
  include_directories(imgui)
  add_subdirectory(imgui)
endif()

find_package(fmt)
find_package(spdlog)
find_package(docopt)

# opencv widget only
add_executable(intro main.cpp)

target_link_libraries(
  intro
  PRIVATE project_options
          project_warnings
          cv
          docopt::docopt
          fmt::fmt
          spdlog::spdlog)

