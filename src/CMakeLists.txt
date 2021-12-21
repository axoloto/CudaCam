
find_package(docopt CONFIG)

add_subdirectory(utils)
add_subdirectory(io)
add_subdirectory(cvp)

if(USE_IMGUI)
  message("Using Dear ImGui")
  add_subdirectory(imgui)
endif()

add_executable(CudaCam main.cpp)
if(USE_IMGUI)
  # imgui app
  target_compile_definitions(CudaCam PRIVATE -DUSE_IMGUI)
  target_link_libraries(CudaCam PRIVATE project_options utils cvProcessing imguiApp docopt::docopt) #project_warnings
else()
  # opencv widget only
  find_package(OPENCV REQUIRED CONFIG)
  target_link_libraries(CudaCam PRIVATE project_options project_warnings utils cvProcessing opencv::opencv_highgui docopt::docopt)
endif()

install(TARGETS CudaCam RUNTIME DESTINATION bin)