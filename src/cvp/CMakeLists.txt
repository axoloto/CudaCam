find_package(CUDAToolkit REQUIRED)
find_package(OPENCV REQUIRED CONFIG)

file(GLOB CPPSRC "*.cpp" "*.hpp")
file(GLOB CUDASRC "cannyEdgeH.cu" "cannyEdgeD.cu")

add_library(cvProcessing STATIC ${CPPSRC} ${CUDASRC})

set_target_properties(cvProcessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cvProcessing PROPERTIES CUDA_ARCHITECTURES "52;61;75")
set_target_properties(cvProcessing PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cvProcessing PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(cvProcessing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cvProcessing PUBLIC project_options utils opencv::opencv_core opencv::opencv_imgproc opencv::opencv_videoio)  #project_warnings

install(TARGETS cvProcessing RUNTIME DESTINATION lib)
