cmake_minimum_required(VERSION 3.5)

project(sdfg_driver
    LANGUAGES
    C
    CXX
    Fortran
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/.dacecache/add/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/.dacecache/add/build)


add_library(sdfg_wrapper STATIC)

target_sources(sdfg_wrapper
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/call_sdfg.f90
)

target_link_libraries(sdfg_wrapper add)
target_link_libraries(sdfg_wrapper dacestub_add)


add_executable(sdfg_driver driver.f90)

target_link_libraries(sdfg_driver
    PUBLIC
    sdfg_wrapper
)


