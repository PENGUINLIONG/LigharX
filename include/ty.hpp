#pragma once
// OptixLab types.
// @PENGUINLIONG
#include <vector>
#include "x.hpp"
#include "except.hpp"

namespace liong {

constexpr size_t L_OPTIMAL_DEVMEM_ALIGN = 128;
// TODO: (penguinliong) dedicated allocation threshold.
constexpr size_t L_DEDICATED_ALLOC_THRESH = 1024 * 1024; // 1MBytes

//
// Core functionalities.
//

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

enum TextureDimension {
  L_TEXTURE_1D,
  L_TEXTURE_2D,
  L_TEXTURE_3D,
};
// Encoded pixel format that can be easily decoded by shift-and ops.
//
//   0bccshibbb
//       \____/
//  `CUarray_format`
//
// - `b`: Number of bits in exponent of 2. Only assigned if its a integral
//   number.
// - `i`: Signedness of integral number.
// - `h`: Is a half-precision floating-point number.
// - `s`: Is a single-precision floating-point number.
// - `c`: Number of texel components (channels) minus 1. Currently only support
//   upto 4 components.
struct PixelFormat {
  union {
    struct {
      uint8_t int_exp2 : 3;
      uint8_t is_signed : 1;
      uint8_t is_half : 1;
      uint8_t is_single : 1;
      uint8_t ncomp : 2;
    };
    uint8_t _raw;
  };
  constexpr PixelFormat(uint8_t raw) : _raw(raw) {}
  constexpr PixelFormat() : _raw() {}
  PixelFormat(const PixelFormat&) = default;
  PixelFormat(PixelFormat&&) = default;
  PixelFormat& operator=(const PixelFormat&) = default;
  PixelFormat& operator=(PixelFormat&&) = default;

  constexpr CUarray_format get_cuda_fmt() const { return (CUarray_format)(_raw & 0b00111111); }
  constexpr uint32_t get_ncomp() const { return ncomp; }

  constexpr bool operator==(const PixelFormat& b) { return _raw == b._raw; }
};
#define L_MAKE_VEC(ncomp, fmt) ((uint8_t)((ncomp - 1) << 6) | (uint8_t)fmt)
constexpr PixelFormat L_FORMAT_UNDEFINED {};
constexpr PixelFormat L_FORMAT_R8_UNORM { L_MAKE_VEC(1, CU_AD_FORMAT_UNSIGNED_INT8) };
constexpr PixelFormat L_FORMAT_R32_SFLOAT { L_MAKE_VEC(1, CU_AD_FORMAT_FLOAT) };
constexpr PixelFormat L_FORMAT_R8G8B8A8_UNORM { L_MAKE_VEC(4, CU_AD_FORMAT_UNSIGNED_INT8) };
constexpr PixelFormat L_FORMAT_R32G32B32A32_SFLOAT { L_MAKE_VEC(4, CU_AD_FORMAT_FLOAT) };
#undef L_MAKE_VEC
struct TextureConfig {
  DeviceMemorySlice tex_slice;
  TextureDimension dim;
  PixelFormat fmt;
  uint32_t w, h, d;
  size_t row_align;
};
struct SamplerConfig {
  CUaddress_mode addr_mode;
  CUfilter_mode filter_mode;
};
struct Texture {
  CUtexObject tex;
  TextureConfig tex_cfg;
};


struct PipelineStageConfig {
  // Name of the stage function in provided module.
  const char* name;
  // Size of data used by a pipeline stage as shader runtime parameter.
  size_t data_size;
};
struct PipelineHitGroupConfig {
  // Intersection stage function name.
  const char* is_name;
  // Any-hit stage function name.
  const char* ah_name;
  // Closest-hit stage function name.
  const char* ch_name;
  // Size of data used by a hit group as shader runtime parameter.
  size_t data_size;
};
struct PipelineCallableConfig {
  // Name of the callable in provided module.
  const char* name;
  // The callable is continuation callable.
  bool is_cc;
  // The callable is direct callable.
  bool is_dc;
  // Size of data used by a callable as shader runtime parameter.
  size_t data_size;
};
// All the data necessary in pipeline createion.
struct PipelineConfig {
  // Whether the pipeline is set up in debug mode. Extra debug information will
  // be provided if enabled.
  bool debug;

  // Pipeline module containing all stages and callables of the pipeline.
  const char* ptx_data;
  size_t ptx_size;

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
  std::vector<PipelineCallableConfig> call_cfgs;
};
struct PipelineLayout {
  size_t sbt_raygen_offset;
  size_t sbt_raygen_stride;

  size_t sbt_except_offset;
  size_t sbt_except_stride;

  size_t sbt_miss_offset;
  size_t sbt_miss_stride;
  size_t nsbt_miss;

  size_t sbt_hitgrp_offset;
  size_t sbt_hitgrp_stride;
  size_t nsbt_hitgrp;

  size_t sbt_call_offset;
  size_t sbt_call_stride;
  size_t nsbt_call;

  size_t sbt_size;
};
// Pipeline-related opaque resources.
struct Pipeline {
  OptixModule mod;
  std::vector<OptixProgramGroup> pgrps;
  OptixPipeline pipe;
  PipelineLayout pipe_layout;
};
// Data to stuff in pipelines as runtime parameters.
struct PipelineData {
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
  // Material buffer allocated in the size specified by the user. The user
  // SHOULD directly write into this buffer to turn in the material data.
  void* mat;
  // Size of the material buffer.
  size_t mat_size;
};



// We need a stable address for asynchronous output.
struct AsFeedback {
  OptixAabb aabb;
  size_t compact_size;
  // Scene object is valid if this field is not null.
  OptixTraversableHandle trav;
  // Acceleration stucture memory.
  DeviceMemory devmem;
};
// Traversable single object.
struct SceneObject {
  // Acceleration structure build feedback.
  AsFeedback* inner;
  // Material data buffer.
  DeviceMemory mat_devmem;
};
// Traversable object collection.
struct Scene {
  // Acceleration structure build feedback.
  AsFeedback* inner;
  // Scene objects as components in the scene
  std::vector<SceneObject> sobjs;
  // Maximal size of an SBT record aligned to `OPTIX_SBT_RECORD_ALIGNMENT`.
  size_t sbt_stride;
};


struct Transaction {
  CUstream stream;
  std::vector<DeviceMemory> mnged_devmems;
  std::vector<void*> mnged_hostmems;
};



namespace type_traits {

template<typename TCont, typename TElem = typename TCont::value_type>
constexpr bool is_buffer_container_v = !std::is_trivially_copyable_v<TCont> &
  std::is_same_v<decltype(TCont::size()), size_t> &
  std::is_trivially_copyable_v<TElem> &
  !std::is_pointer_v<decltype(TCont::data)>;

template<typename T>
constexpr bool is_buffer_object_v = std::is_trivially_copyable_v<T> &
  !std::is_pointer_v<T>;

}





//
// Extensions.
//

namespace ext {

struct NaivePipelineConfig {
  bool debug;
  const char* ptx_data;
  size_t ptx_size;
  unsigned trace_depth;
  const char* rg_name;
  const char* ms_name;
  const char* ah_name;
  const char* ch_name;
};



}


}
