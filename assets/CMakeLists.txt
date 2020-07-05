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
        COMMAND ${NVCC} -ptx --use_fast_math --expt-relaxed-constexpr --generate-line-info -I"${CUDA_TOOLKIT_ROOT_DIR}/include" -I"${OptiX_INCLUDE}" -I"${PROJECT_SOURCE_DIR}/include" -o ${SHADER_DST} ${SHADER_SRC}
        COMMENT "compiling shader: ${SHADER_SRC}"
        VERBATIM
    )
    add_custom_target(PTX_${SHADER_SRC_NAME} DEPENDS ${SHADER_DST})
    add_dependencies(OptixLab PTX_${SHADER_SRC_NAME})
endforeach()
