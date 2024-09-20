cmake_minimum_required(VERSION 3.29)
project(psycheC_ast)

set(CMAKE_CXX_STANDARD 17)

# Include directories for Psyche-C headers
include_directories("/Users/richardcernansky/Desktop/bakalarka/psychec/C")

# Add the path to the Psyche-C dynamic libraries
link_directories("/Users/richardcernansky/Desktop/bakalarka/psychec/C")

link_directories("/Users/richardcernansky/Desktop/bakalarka/psychec/common")

# Add the executable
add_executable(psycheC_ast main.cpp
        AnalysisVisitor.cpp
        AnalysisVisitor.h)

# Link the Psyche-C libraries
target_link_libraries(psycheC_ast PRIVATE psychecfe psychecstd psychecommon)
