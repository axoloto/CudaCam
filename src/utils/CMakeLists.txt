find_package(fmt)
find_package(spdlog)

file(GLOB SRC "*.cpp" "*.hpp")

add_library(utils ${SRC})

# Not sure why it is not given through spdlog findPackage
target_compile_definitions(utils PUBLIC -DSPDLOG_FMT_EXTERNAL)

target_compile_definitions(utils PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(utils PUBLIC -DINSTALL_DIR="${CMAKE_INSTALL_DIR}")
target_compile_definitions(utils PUBLIC -DVERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
target_compile_definitions(utils PUBLIC -DVERSION_MINOR="${PROJECT_VERSION_MINOR}")
target_compile_definitions(utils PUBLIC -DVERSION_PATCH="${PROJECT_VERSION_PATCH}")
message(STATUS "CMAKE Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE Install Directory: ${CMAKE_INSTALL_DIR}")

target_include_directories(utils PUBLIC ${fmt_INCLUDE_DIR} ${spdlog_INCLUDE_DIR})

target_link_libraries(utils PRIVATE fmt::fmt spdlog::spdlog)