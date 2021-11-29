# imgui test executable, with full warnings enabled
find_package(SDL2 REQUIRED)
find_package(imgui REQUIRED)
find_package(glad REQUIRED)
find_package(OpenGL REQUIRED)

file(GLOB SRC "*.cpp" "*.hpp" ${imgui_RES_DIRS}/bindings/imgui_impl_opengl3.cpp ${imgui_RES_DIRS}/bindings/imgui_impl_sdl.cpp)

add_library(imguiApp ${SRC})

target_compile_definitions(imguiApp PUBLIC -DIMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(imguiApp PUBLIC
                                    ${CMAKE_CURRENT_SOURCE_DIR}
                                    ${imgui_INCLUDE_DIR}
                                    ${imgui_RES_DIRS}
                                    ${glad_INCLUDE_DIR}
                                    ${SDL2_INCLUDE_DIRS}
                                    ${OPENGL_INCLUDE_DIRS})

target_link_libraries(imguiApp PRIVATE project_options utils cv imgui::imgui ${glad_LIBRARIES} ${SDL2_LIBRARIES}) #project_warnings
