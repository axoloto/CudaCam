
find_package(docopt)

add_subdirectory(utils)
add_subdirectory(cv)

# imgui app
option(USE_IMGUI "Enable compilation of ImGui sample" ON)
if(USE_IMGUI)
  message("Using Dear ImGui")
  add_subdirectory(imgui)
endif()

# opencv widget only
add_executable(cudaCam main.cpp)

if(USE_IMGUI)
  target_compile_definitions(cudaCam PUBLIC -DUSE_IMGUI)
  #target_include_directories(cudaCam PRIVATE imgui cv)
  target_link_libraries(cudaCam PRIVATE project_options utils cv imguiApp docopt::docopt) #project_warnings
else()
  target_include_directories(cudaCam PRIVATE utils cv)
  target_link_libraries(cudaCam PRIVATE project_options project_warnings utils cv docopt::docopt)
endif()
