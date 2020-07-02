#pragma once
// Extensions to core functionalities.
// @PENGUINLIONG
#include <core.hpp>

namespace liong {

namespace ext {

// Create a naive pipeline that is enough to do simple ray tracing tasks. A
// naive pipline is a predefined cofiguration with `lauch_param_name` defaulted
// to `cfg`, `npayload_wd` and `nattr_wd` defaulted to 2 (to carry a pointer).
Pipeline create_native_pipe(
  const Context& ctxt,
  const NaivePipelineConfig& naive_pipe_cfg
);

inline DeviceMemorySlice slice_naive_pipe_mat(
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  uint32_t idx
) {
  return slice_pipe_data(pipe, pipe_data, OPTIX_PROGRAM_GROUP_KIND_HITGROUP, idx);
}
inline DeviceMemorySlice slice_naive_pipe_env(
  const Pipeline& pipe,
  const PipelineData& pipe_data
) {
  return slice_pipe_data(pipe, pipe_data, OPTIX_PROGRAM_GROUP_KIND_MISS, 0);
}


}

}
