find_package(OPENCV REQUIRED CONFIG)

file(GLOB SRC "*.cpp" "*.hpp")

add_library(io STATIC ${SRC})

target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(io PUBLIC project_options utils opencv::opencv_core opencv::opencv_imgproc opencv::opencv_videoio)  #project_warnings

install(TARGETS io RUNTIME DESTINATION lib)
