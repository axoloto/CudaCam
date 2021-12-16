find_package(fmt CONFIG)
find_package(spdlog CONFIG)

file(GLOB SRC "*.cpp" "*.hpp")

add_library(utils ${SRC})

# Not sure why it is not given through spdlog findPackage
target_compile_definitions(utils PUBLIC -DSPDLOG_FMT_EXTERNAL)

target_compile_definitions(utils PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(utils PUBLIC -DINSTALL_DIR="${CMAKE_INSTALL_DIR}")
target_compile_definitions(utils PUBLIC -DVERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
target_compile_definitions(utils PUBLIC -DVERSION_MINOR="${PROJECT_VERSION_MINOR}")
target_compile_definitions(utils PUBLIC -DVERSION_PATCH="${PROJECT_VERSION_PATCH}")

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utils PUBLIC fmt::fmt spdlog::spdlog)