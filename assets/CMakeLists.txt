# Compile OptixLabShaders.
file(GLOB SHADER_SRCS "${PROJECT_SOURCE_DIR}/assets/*.cu")
set(CUDA_NVCC_FLAGS --use_fast_math --expt-relaxed-constexpr --generate-line-info)
cuda_include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include" "${OptiX_INCLUDE}" "${PROJECT_SOURCE_DIR}/include")
foreach(SHADER_SRC IN ITEMS ${SHADER_SRCS})
	# Please DO NOT differentiate names using extension version numbers like
	# `demo.1.cu` and `demo.2.cu`. CMake is not so smart to support excluding
	# only the last extension name. Very sad. I mean, I use it a lot.
	get_filename_component(SHADER_SRC_NAME ${SHADER_SRC} NAME_WE)
	cuda_compile_ptx(PTX_OUT ${SHADER_SRC})
	add_custom_target(PTX_${SHADER_SRC_NAME} DEPENDS ${PTX_OUT})
	add_dependencies(OptixLab PTX_${SHADER_SRC_NAME})
endforeach()
