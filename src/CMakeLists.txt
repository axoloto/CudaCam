option(USE_IMGUI "Enable compilation of ImGui sample" ON)

find_package(docopt)

add_subdirectory(utils)
add_subdirectory(cv)

# imgui app
if(USE_IMGUI)
  message("Using Dear ImGui")
  add_subdirectory(imgui)
endif()

# opencv widget only
add_executable(intro main.cpp)

target_include_directories(intro PRIVATE utils cv)

target_link_libraries(intro PRIVATE project_options project_warnings utils cv docopt::docopt)

