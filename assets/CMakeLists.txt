# Compile OptixLabShaders.
file(GLOB SHADER_SRCS "${PROJECT_SOURCE_DIR}/assets/*.cu")
set(CUDA_NVCC_FLAGS --use_fast_math --expt-relaxed-constexpr --generate-line-info)
CUDA_INCLUDE_DIRECTORIES("${CUDA_TOOLKIT_ROOT_DIR}/include" "${OptiX_INCLUDE}" "${PROJECT_SOURCE_DIR}/include")
cuda_compile_ptx(PTXS ${SHADER_SRCS})
add_custom_target(PTX_${SHADER_SRC_NAME} DEPENDS ${PTXS})
add_dependencies(OptixLab PTX_${SHADER_SRC_NAME})
