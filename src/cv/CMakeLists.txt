find_package(CUDAToolkit REQUIRED)
find_package(OPENCV REQUIRED CONFIG)

add_library(cvCuda STATIC "cannyEdge.cu" "cannyEdge.cuh")

set_target_properties(cvCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cvCuda PROPERTIES CUDA_ARCHITECTURES "52;61;75")
set_target_properties(cvCuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cvCuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

file(GLOB CPPSRC "*.cpp" "*.hpp")

add_library(cv STATIC ${CPPSRC})

target_include_directories(cv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cv PUBLIC project_options utils opencv::opencv_core opencv::opencv_imgproc opencv::opencv_videoio PRIVATE cvCuda)  #project_warnings
