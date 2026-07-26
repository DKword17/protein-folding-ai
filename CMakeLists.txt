cmake_minimum_required(VERSION 3.18)
project(kinetics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(pybind11 REQUIRED)
set(KINETICS_SOURCES
    kinetics/types.h kinetics/verlet.h kinetics/verlet.cpp
    kinetics/langevin.h kinetics/langevin.cpp kinetics/module.cpp)
pybind11_add_module(_kinetics_core MODULE ${KINETICS_SOURCES})
target_include_directories(_kinetics_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kinetics)
install(TARGETS _kinetics_core LIBRARY DESTINATION kinetics RUNTIME DESTINATION kinetics)
