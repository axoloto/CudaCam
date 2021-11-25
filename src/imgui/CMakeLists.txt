# imgui test executable, with full warnings enabled
find_package(SDL2 REQUIRED)
find_package(imgui REQUIRED)

file(GLOB SRC "*.cpp" "*.hpp" ${imgui_RES_DIRS}/bindings/imgui_impl_opengl3.cpp ${imgui_RES_DIRS}/bindings/imgui_impl_sdl.cpp)

add_executable(test_imgui ${SRC})

target_include_directories(test_imgui PRIVATE ${imgui_RES_DIRS} ${SDL2_INCLUDE_DIRS})

target_link_libraries(test_imgui PRIVATE cv project_warnings project_options imgui::imgui ${SDL2_LIBRARIES})
