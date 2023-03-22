cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#[=[
CMakeLists.txt
=================

Note this is a bit laborious as does not use the 
Opticks CMake machinery 

#]=]


set(name C4)
project(${name})

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"

find_package(Geant4 CONFIG)   

if(Geant4_FOUND)
    message(STATUS "Geant4_FOUND   : ${Geant4_FOUND} ")
    message(STATUS "Geant4_VERSION : ${Geant4_VERSION} ")
    message(STATUS "Geant4_LIBRARIES : ${Geant4_LIBRARIES} ")
    message(STATUS "Geant4_INCLUDE_DIR : ${Geant4_INCLUDE_DIR} ")
else()
    message(STATUS "Geant4 not found")
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
install(TARGETS ${name}  LIBRARY DESTINATION lib)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


