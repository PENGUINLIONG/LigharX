#pragma once
// OptixLab types.
// @PENGUINLIONG
#include <vector>
#include "x.hpp"
#include "except.hpp"

namespace liong {


struct Context {
  CUdevice dev;
  CUcontext cuda_ctxt;
  OptixDeviceContext optix_dc;
};



struct DeviceMemorySlice {
  CUdeviceptr ptr;
  size_t size;
};
struct DeviceMemory {
  // Base address of memory allocation.
  CUdeviceptr alloc_base;
  // Base address pointer aligned to specified alignment.
  CUdeviceptr ptr;
  // Size of (total) allocated memory including alignment padding.
  size_t alloc_size;
  // Size of allocated memory aligned to specified alignment.
  size_t size;

  inline operator DeviceMemorySlice() const {
    return DeviceMemorySlice { ptr, size };
  }
  inline DeviceMemorySlice slice(size_t offset, size_t size) const {
    ASSERT << (offset + size <= this->size)
      << "slice out of range";
    return DeviceMemorySlice { ptr + offset, size };
  }
  inline DeviceMemorySlice slice(size_t offset) const {
    return slice(offset, size - offset);
  }
};



struct PipelineStageConfig {
  // Name of the stage function in provided module.
  const char* name;
  // Data used in a pipeline stage which will be copied to device memory.
  const void* data;
  // Size of `data` in bytes.
  size_t size;
};
struct PipelineHitGroupConfig {
  // Intersection stage function name.
  const char* is_name;
  // Any-hit stage function name.
  const char* ah_name;
  // Closest-hit stage function name.
  const char* ch_name;
  // Data used by a hit group which will be copied to device memory.
  const void* data;
  // Size of `data` in bytes.
  size_t size;
};
// All the data necessary in pipeline createion.
struct PipelineConfig {
  // Whether the pipeline is set up in debug mode. Extra debug information will
  // be provided if enabled.
  bool debug;

  // Path to the pipeline module containing all stages and callables of the
  // pipeline.
  const char* mod_path;
  // Launch parameter variable name. The parameter variable will be ignored if
  // this field is empty.
  const char* launch_param_name;
  // Number of words used for the payload. [0..8]
  int npayload_wd;
  // Number of words used for the attributes. [0..8]
  int nattr_wd;
  // Maximum trace recursion depth. [0..31]
  unsigned trace_depth;

  // Although it's POSSIBLE to use multiple raygen functions but for efficiency
  // here we DO NOT support it.
  PipelineStageConfig rg_cfg;
  PipelineStageConfig ex_cfg;
  std::vector<PipelineStageConfig> ms_cfgs;
  std::vector<PipelineHitGroupConfig> hitgrp_cfgs;
  std::vector<PipelineStageConfig> dc_cfgs;
  std::vector<PipelineStageConfig> cc_cfgs;
};
// Pipeline-related opaque resources.
struct Pipeline {
  OptixModule mod;
  std::vector<OptixProgramGroup> pgrps;
  OptixPipeline pipe;
  OptixShaderBindingTable sbt;
  DeviceMemory sbt_devmem;
};



struct Framebuffer {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  DeviceMemory framebuf_devmem;
};



struct MeshConfig {
  // Vertex data buffer.
  const void* vert_buf;
  // Format of vertex data.
  OptixVertexFormat vert_fmt;
  // Number of vertices in the buffer.
  size_t nvert;
  // Stride between packs of vertex data.
  size_t vert_stride;

  // Index data buffer.
  const void* idx_buf;
  // Format of index tuple.
  OptixIndicesFormat idx_fmt;
  // Number of triangles in `idx_buf`.
  size_t ntri;
  // Stride between tuples of triangle vertex indices.
  size_t tri_stride;
};
struct Mesh {
  DeviceMemory devmem;
  DeviceMemorySlice vert_slice;
  DeviceMemorySlice idx_slice;
  OptixBuildInput build_in;
};



struct SceneObject {
  bool dirty;
  OptixTraversableHandle trav;
  OptixAabb aabb;
  // Vertex and index buffer. Vertex data is placed at first and index follows
  // that.
  DeviceMemory devmem;
};



struct Transaction {
  CUstream stream;
  DeviceMemory lparam_devmem;
};


}
