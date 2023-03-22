cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name C4Test)
project(${name} VERSION 0.1.0)

set(TEST_SOURCES
     C4OpBoundaryProcessTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} C4)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




