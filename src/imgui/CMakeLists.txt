# imgui test executable, with full warnings enabled
find_package(SDL2 REQUIRED CONFIG)
find_package(imgui REQUIRED CONFIG)
find_package(glad REQUIRED CONFIG)
find_package(OpenGL REQUIRED)

string(TOUPPER ${CMAKE_BUILD_TYPE} SUFFIX)
set(IMGUI_RES_PATH "${imgui_RES_DIRS_${SUFFIX}}/bindings")

file(GLOB SRC "*.cpp" "*.hpp" ${IMGUI_RES_PATH}/imgui_impl_opengl3.cpp ${IMGUI_RES_PATH}/imgui_impl_sdl.cpp)

add_library(imguiApp ${SRC})

target_compile_definitions(imguiApp PUBLIC -DIMGUI_IMPL_OPENGL_LOADER_GLAD)
target_include_directories(imguiApp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${IMGUI_RES_PATH})

target_link_libraries(imguiApp PUBLIC cv glad::glad SDL2::SDL2 PRIVATE project_options utils imgui::imgui) #project_warnings
