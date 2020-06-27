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

# Build the OptixLab.
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    ${OptiX_INCLUDE}
    ${CUDA_TOOLKIT_INCLUDE}
)
file(GLOB SRCS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(OptixLab ${SRCS})

# Compile OptixLabShaders.
file(GLOB SHADER_SRCS "${PROJECT_SOURCE_DIR}/assets/*.cu")
find_program(NVCC nvcc REQUIRED)
foreach(SHADER_SRC IN ITEMS ${SHADER_SRCS})
    get_filename_component(SHADER_SRC_DIR ${SHADER_SRC} DIRECTORY)
    get_filename_component(SHADER_SRC_NAME ${SHADER_SRC} NAME_WE)
    make_directory(${CMAKE_BINARY_DIR}/assets)
    set(SHADER_DST "${CMAKE_BINARY_DIR}/assets/${SHADER_SRC_NAME}.ptx")
    add_custom_command(
        OUTPUT ${SHADER_DST}
        DEPENDS ${SHADER_SRC}
        COMMAND ${NVCC} -ptx --use_fast_math --generate-line-info -I"${CUDA_TOOLKIT_ROOT_DIR}/include" -I"${OptiX_INCLUDE}" -I"${PROJECT_SOURCE_DIR}/include" -o ${SHADER_DST} ${SHADER_SRC}
        COMMENT "compiling shader: ${SHADER_SRC}"
        VERBATIM
    )
    add_custom_target(PTX_${SHADER_SRC_NAME} DEPENDS ${SHADER_DST})
    add_dependencies(OptixLab PTX_${SHADER_SRC_NAME})
endforeach()

# Link things up.
target_link_libraries(
    OptixLab
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
