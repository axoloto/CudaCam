# imgui test executable, with full warnings enabled
find_package(SDL2 REQUIRED)
find_package(imgui REQUIRED)

file(GLOB SRC "*.cpp" "*.hpp" ${imgui_RES_DIRS}/bindings/imgui_impl_opengl3.cpp ${imgui_RES_DIRS}/bindings/imgui_impl_sdl.cpp)

add_library(imguiApp ${SRC})

target_include_directories(imguiApp PUBLIC ${CMAKE_SOURCE_DIR}/src/utils ${CMAKE_SOURCE_DIR}/src/cv ${imgui_INCLUDE_DIR} ${imgui_RES_DIRS} ${SDL2_INCLUDE_DIRS})

target_link_libraries(imguiApp PRIVATE project_options utils cv imgui::imgui ${SDL2_LIBRARIES}) #project_warnings
