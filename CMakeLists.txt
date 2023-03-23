cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#[=[
CMakeLists.txt
=================

Note this is a bit laborious as does not use the 
Opticks CMake machinery 

Following primitive non-target based CMake approach for consistency with junosw
Potential for troubles when used with Opticks target based CMake.  

Need to derive a C4Config.cmake

#]=]


set(name Custom4)
project(${name})
set(${PROJECT_NAME}_VERSION_MAJOR 0) 
set(${PROJECT_NAME}_VERSION_MINOR 0)  
set(${PROJECT_NAME}_VERSION_PATCH 4)  
set(${PROJECT_NAME}_VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}")


include(GNUInstallDirs)
#set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"


set(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH})


find_package(Geant4 REQUIRED)   

get_filename_component(Custom4_Geant4_PREFIX "${Geant4_INCLUDE_DIR}/../.." ABSOLUTE) ## DUH : Geant4 does not provide PREFIX
find_program(Custom4_GEANT4_CONFIG_BIN geant4-config PATHS ${C4_Geant4_PREFIX}/bin)
execute_process(
     COMMAND ${Custom4_GEANT4_CONFIG_BIN} --cflags
     OUTPUT_VARIABLE Custom4_CFLAGS                                   ## the cflags has the correct include directory 
     OUTPUT_STRIP_TRAILING_WHITESPACE
    )

string(REPLACE "-std=c++11" "" Custom4_CFLAGS ${Custom4_CFLAGS})

execute_process(
    COMMAND ${Custom4_GEANT4_CONFIG_BIN} --libs
    OUTPUT_VARIABLE Custom4_GEANT4_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions(${Custom4_CFLAGS})


set(Custom4_VERBOSE YES)
if(Custom4_VERBOSE)
    message(STATUS "Custom4_VERBOSE      : ${Custom4_VERBOSE} ")
    message(STATUS "Geant4_FOUND         : ${Geant4_FOUND} ")
    message(STATUS "Geant4_VERSION       : ${Geant4_VERSION} ")
    message(STATUS "Geant4_INCLUDE_DIRS  : ${Geant4_INCLUDE_DIRS} ")
    message(STATUS "Geant4_DEFINITIONS   : ${Geant4_DEFINITIONS} ")
    message(STATUS "Geant4_LIBRARIES     : ${Geant4_LIBRARIES} ")
    message(STATUS "Geant4_BUILD_TYPE    : ${Geant4_BUILD_TYPE} ")
    message(STATUS "Geant4_CXX_FLAGS     : ${Geant4_CXX_FLAGS} (recommended)")
    message(STATUS "Geant4_CXXSTD        : ${Geant4_CXXSTD}  (standard used to compile Geant4) ")
    message(STATUS "Geant4_TLS_MODEL     : ${Geant4_TLS_MODEL}  (only set if install has multithreading support)")

    message(STATUS "Geant4_INCLUDE_DIR    : ${Geant4_INCLUDE_DIR}  (expected ending: include/Geant4) ")
    message(STATUS "Custom4_Geant4_PREFIX : ${Custom4_Geant4_PREFIX}      (derived from INCLUDE_DIR) ")
    message(STATUS "Custom4_GEANT4_CONFIG_BIN : ${Custom4_GEANT4_CONFIG_BIN} ")
    message(STATUS "Custom4_CFLAGS            : ${Custom4_CFLAGS}        (from geant4-config --cflags : with -std=c++11 removed) ")
    message(STATUS "Custom4_GEANT4_LIBRARIES  : ${Custom4_GEANT4_LIBRARIES}  (from geant4-config --libs) ")

endif()


set(SOURCES 
    C4OpBoundaryProcess.cc
)

set(HEADERS
    C4OpBoundaryProcess.hh
    C4IPMTAccessor.h
    C4CustomART.h
    C4CustomART_Debug.h
    C4MultiLayrStack.h
    C4Touchable.h
)

add_library( ${name} SHARED ${SOURCES} ${HEADERS} )

target_include_directories( ${name} PUBLIC 
      $<BUILD_INTERFACE:${Geant4_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)
target_link_libraries(      ${name} PUBLIC ${Geant4_LIBRARIES})

set(CUSTOM4_RELATIVE_LIBDIR  ${CMAKE_INSTALL_LIBDIR})
set(CUSTOM4_RELATIVE_INCDIR  ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(TARGETS ${name}  LIBRARY DESTINATION ${CUSTOM4_RELATIVE_LIBDIR})
install(FILES ${HEADERS}  DESTINATION        ${CUSTOM4_RELATIVE_INCDIR})

set(CUSTOM4_CMAKE_DIR  ${CUSTOM4_RELATIVE_LIBDIR}/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION})


set(CUSTOM4_SETUP "
# CUSTOM4_SETUP 

get_filename_component(Custom4_CMAKE_PATH   \"\${CMAKE_CURRENT_LIST_FILE}\"    ABSOLUTE)
get_filename_component(Custom4_CMAKE_DIR    \"\${CMAKE_CURRENT_LIST_FILE}/..\" ABSOLUTE)
get_filename_component(Custom4_LIBRARY_DIR  \"\${CMAKE_CURRENT_LIST_FILE}/../..\" ABSOLUTE)
get_filename_component(Custom4_PREFIX       \"\${CMAKE_CURRENT_LIST_FILE}/../../..\" ABSOLUTE)
get_filename_component(Custom4_INCLUDE_DIR  \"\${Custom4_PREFIX}/${CUSTOM4_RELATIVE_INCDIR}\" ABSOLUTE)

set(Custom4_INCLUDE_DIRS \${Custom4_INCLUDE_DIR})
set(Custom4_LIBRARIES \"-L\${Custom4_LIBRARY_DIR} -l${PROJECT_NAME}\" )
set(Custom4_DEFINITIONS -DWITH_CUSTOM4 )

")


configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/Templates/Custom4Config.cmake.in
  ${PROJECT_BINARY_DIR}/Custom4Config.cmake
  @ONLY
  )

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/Templates/Custom4ConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/Custom4ConfigVersion.cmake
  @ONLY
  )

install(FILES
  ${PROJECT_BINARY_DIR}/Custom4Config.cmake
  ${PROJECT_BINARY_DIR}/Custom4ConfigVersion.cmake
  DESTINATION ${CUSTOM4_CMAKE_DIR}
  )

