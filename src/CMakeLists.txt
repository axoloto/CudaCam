
find_package(docopt CONFIG)

add_subdirectory(utils)
add_subdirectory(cv)

option(USE_IMGUI "Enable compilation of ImGui sample" ON)
if(USE_IMGUI)
  message("Using Dear ImGui")
  add_subdirectory(imgui)
endif()

add_executable(cudaCam main.cpp)

if(USE_IMGUI)
  # imgui app
  target_compile_definitions(cudaCam PRIVATE -DUSE_IMGUI)
  target_link_libraries(cudaCam PRIVATE project_options utils cv imguiApp docopt::docopt) #project_warnings
else()
  # opencv widget only
  target_link_libraries(cudaCam PRIVATE project_options project_warnings utils cv docopt::docopt)
endif()
