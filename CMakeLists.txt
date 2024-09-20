cmake_minimum_required(VERSION 3.29)
project(psycheC_ast)

set(CMAKE_CXX_STANDARD 17)

# Find the parent directory of the project (one level up)
set(PARENT_DIR ${CMAKE_SOURCE_DIR}/..)

# Include directories for Psyche-C headers using relative path
include_directories(${PARENT_DIR}/psychec/C)

# Link directories for Psyche-C dynamic libraries using relative path
link_directories(${PARENT_DIR}/psychec/C)
link_directories(${PARENT_DIR}/psychec/common)

# Add the executable
add_executable(psycheC_ast main.cpp
        AnalysisVisitor.cpp
        AnalysisVisitor.h)

# Link the Psyche-C libraries
target_link_libraries(psycheC_ast PRIVATE psychecfe psychecstd psychecommon)

