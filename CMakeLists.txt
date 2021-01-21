cmake_minimum_required(VERSION 2.8.12)
project(neurogen)
# Set source directory
set(SOURCE_DIR "src/neurogen")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})

add_subdirectory(lib/pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(lib/draco ${CMAKE_BINARY_DIR}/draco_build)
include_directories("${CMAKE_BINARY_DIR}" lib/draco/src)

pybind11_add_module(backend MODULE ${SOURCES} "${SOURCE_DIR}/bindings.cc")
target_link_libraries(backend PRIVATE draco_static)