cmake_minimum_required (VERSION 3.12)

project ("OptixLab" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -use_fast_math" )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED VERSION 7.0)
find_package(Assimp)

if(assimp_FOUND)
    add_definitions(-DL_USE_ASSIMP=1)
endif()

# Build the OptixLab.
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    ${OptiX_INCLUDE}
    ${CUDA_TOOLKIT_INCLUDE}
    ${ASSIMP_INCLUDE_DIR}
)
file(GLOB SRCS "${PROJECT_SOURCE_DIR}/src/*")
file(GLOB INCS "${PROJECT_SOURCE_DIR}/include/*")
add_library(OptixLab ${SRCS} ${INCS})

# Link things up.
target_link_libraries(
    OptixLab
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${ASSIMP_LIBRARY}
)

add_subdirectory("${PROJECT_SOURCE_DIR}/examples")
