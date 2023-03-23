cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Custom4FindPackageExample)
project(${name})

find_package(Custom4 REQUIRED)

message(STATUS "Custom4_FOUND         : ${Custom4_FOUND} ")
message(STATUS "Custom4_VERSION       : ${Custom4_VERSION} ")
message(STATUS "Custom4_PREFIX        : ${Custom4_PREFIX} ")
message(STATUS "Custom4_INCLUDE_DIR   : ${Custom4_INCLUDE_DIR} ")
message(STATUS "Custom4_INCLUDE_DIRS  : ${Custom4_INCLUDE_DIRS} ")
message(STATUS "Custom4_DEFINITIONS   : ${Custom4_DEFINITIONS} ")
message(STATUS "Custom4_LIBRARY_DIR   : ${Custom4_LIBRARY_DIR} ")
message(STATUS "Custom4_LIBRARIES     : ${Custom4_LIBRARIES} ")
message(STATUS "Custom4_CMAKE_PATH    : ${Custom4_CMAKE_PATH} ")
message(STATUS "Custom4_CMAKE_DIR     : ${Custom4_CMAKE_DIR} ")





