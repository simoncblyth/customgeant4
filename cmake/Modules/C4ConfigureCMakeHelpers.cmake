#[=[
C4ConfigureCMakeHelpers.cmake
===============================

Following Geant4 this module configures and installs CMake modules 
allowing clients to find and use C4 libraries using CMake's find_package command.

#]=]


set(CUSTOM4_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}-${${PROJECT_NAME}_VERSION})
set(CUSTOM4_RELATIVE_HEADER_PATH include/${PROJECT_NAME})

set(CUSTOM4_SETUP "
# Geant4 configured for the install with relative paths, so use these
get_filename_component(Custom4_INCLUDE_DIR \"\${Custom4_PREFIX}/${CUSTOM4_RELATIVE_HEADER_PATH}\" ABSOLUTE)

set(Custom4_DEFINITIONS -DWITH_CUSTOM4 )
set(Custom4_INCLUDE_DIRS \${Custom4_INCLUDE_DIR})
set(Custom4_LIBRARY_DIR \${Custom4_PREFIX}/lib)
set(Custom4_LIBRARIES \"-L\${Custom4_LIBRARY_DIR} -l${PROJECT_NAME}\" )

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

