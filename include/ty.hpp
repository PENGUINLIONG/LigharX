#pragma once
// LigharX types.
// @PENGUINLIONG
#include <vector>
#include <stdexcept>
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

  inline DeviceMemorySlice slice(size_t offset, size_t size) const {
    ASSERT << (offset + size <= this->size)
      << "slice out of range";
    return DeviceMemorySlice { ptr + offset, size };
  }
  inline DeviceMemorySlice slice(size_t offset) const {
    return slice(offset, size - offset);
  }
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
  constexpr uint32_t get_ncomp() const { return ncomp + 1; }
  constexpr uint32_t get_fmt_size() const {
    // TODO: (penguinliong) Ensure it still matches when more formats are accepted
    // by cuda.
    uint32_t comp_size = 0;
    if (is_single) {
      comp_size = sizeof(float);
    } else if (is_half) {
      comp_size = sizeof(uint16_t);
    } else {
      comp_size = 4 << (int_exp2) >> 3;
    }
    return get_ncomp() * comp_size;
  }
  // Extract a real number from the buffer.
  inline float extract(const void* buf, size_t i, uint32_t comp) const {
    if (comp > ncomp) { return 0.f; }
    if (is_single) {
      return ((const float*)buf)[i * get_ncomp() + comp];
    } else if (is_half) {
      throw std::logic_error("not implemented yet");
    } else if (is_signed) {
      switch (int_exp2) {
      case 1:
        return ((const int8_t*)buf)[i * get_ncomp() + comp] / 128.f;
      case 2:
        return ((const int16_t*)buf)[i * get_ncomp() + comp] / 32768.f;
      case 3:
        return ((const int32_t*)buf)[i * get_ncomp() + comp] / 2147483648.f;
      }
    } else {
      switch (int_exp2) {
      case 1:
        return ((const uint8_t*)buf)[i * get_ncomp() + comp] / 255.f;
      case 2:
        return ((const uint16_t*)buf)[i * get_ncomp() + comp] / 65535.f;
      case 3:
        return ((const uint32_t*)buf)[i * get_ncomp() + comp] / 4294967296.f;
      }
    }
    ASSERT << false
      << "unsupported framebuffer format";
  }
  // Extract a 32-bit word from the buffer as an integer. If the format is
  // shorter than 32 bits zeros are padded from MSB.
  inline uint32_t extract_word(const void* buf, size_t i, uint32_t comp) const {
    ASSERT << (!is_single & !is_half)
      << "real number type cannot be extracted as bitfield";
    switch (int_exp2) {
      case 1:
        return ((const uint8_t*)buf)[i * get_ncomp() + comp];
      case 2:
        return ((const uint16_t*)buf)[i * get_ncomp() + comp];
      case 3:
        return ((const uint32_t*)buf)[i * get_ncomp() + comp];
    }
  }
  constexpr bool operator==(const PixelFormat& b) { return _raw == b._raw; }
};
#define L_MAKE_VEC(ncomp, fmt)                                                 \
  ((uint8_t)((ncomp - 1) << 6) | (uint8_t)fmt)
#define L_DEF_FMT(name, ncomp, fmt)                                            \
  constexpr PixelFormat L_FORMAT_##name { L_MAKE_VEC(ncomp, fmt) }
L_DEF_FMT(UNDEFINED, 0, 0);

L_DEF_FMT(R8_UNORM,            1, CU_AD_FORMAT_UNSIGNED_INT8);
L_DEF_FMT(R8G8_UNORM,          2, CU_AD_FORMAT_UNSIGNED_INT8);
L_DEF_FMT(R8G8B8_UNORM,        3, CU_AD_FORMAT_UNSIGNED_INT8);
L_DEF_FMT(R8G8B8A8_UNORM,      4, CU_AD_FORMAT_UNSIGNED_INT8);

L_DEF_FMT(R16_UINT,            1, CU_AD_FORMAT_UNSIGNED_INT16);
L_DEF_FMT(R16G16_UINT,         2, CU_AD_FORMAT_UNSIGNED_INT16);
L_DEF_FMT(R16G16B16_UINT,      3, CU_AD_FORMAT_UNSIGNED_INT16);
L_DEF_FMT(R16G16B16A16_UINT,   4, CU_AD_FORMAT_UNSIGNED_INT16);

L_DEF_FMT(R32_UINT,            1, CU_AD_FORMAT_UNSIGNED_INT32);
L_DEF_FMT(R32G32_UINT,         2, CU_AD_FORMAT_UNSIGNED_INT32);
L_DEF_FMT(R32G32B32_UINT,      3, CU_AD_FORMAT_UNSIGNED_INT32);
L_DEF_FMT(R32G32B32A32_UINT,   4, CU_AD_FORMAT_UNSIGNED_INT32);

L_DEF_FMT(R32_SFLOAT,          1, CU_AD_FORMAT_FLOAT);
L_DEF_FMT(R32G32_SFLOAT,       2, CU_AD_FORMAT_FLOAT);
L_DEF_FMT(R32G32B32_SFLOAT,    3, CU_AD_FORMAT_FLOAT);
L_DEF_FMT(R32G32B32A32_SFLOAT, 4, CU_AD_FORMAT_FLOAT);
#undef L_DEF_FMT
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
  const char* launch_cfg_name;
  // Number of words used for the payload. [0..8]
  int npayload_wd;
  // Number of words used for the attributes. [0..8]
  int nattr_wd;
  // Maximum trace recursion depth. [0..31]
  unsigned trace_depth;
  // Maximum number of instances can be contained in the traversed scene; MUST
  // be equal to or greater than 1.
  uint32_t max_ninst;

  // Although it's POSSIBLE to use multiple raygen functions but for efficiency
  // here we DO NOT support it.
  PipelineStageConfig rg_cfg;
  PipelineStageConfig ex_cfg;
  std::vector<PipelineStageConfig> ms_cfgs;
  std::vector<PipelineHitGroupConfig> hitgrp_cfgs;
  std::vector<PipelineCallableConfig> call_cfgs;

  size_t launch_cfg_size;
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
  uint32_t max_ninst;

  size_t launch_cfg_size;
};
// Pipeline-related opaque resources.
struct Pipeline {
  OptixModule mod;
  std::vector<OptixProgramGroup> pgrps;
  OptixPipeline pipe;
  PipelineLayout pipe_layout;
};
// Data to stuff in pipelines as runtime parameters. A single pipeline can be
// used to traverse multiple scene (objects) with respective pipeline data.
struct PipelineData {
  OptixShaderBindingTable sbt;
  DeviceMemory devmem;
  DeviceMemorySlice sbt_devmem;
  // TODO: (penguinliong) Append this to the end of `sbt_devmem`.
  DeviceMemorySlice launch_cfg_devmem;
};



struct Framebuffer {
  PixelFormat fmt;
  uint3 dim;
  DeviceMemory framebuf_devmem;
};



struct MeshConfig {
  // Pre-applied transformation of mesh.
  Transform pretrans;

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
  DeviceMemorySlice pretrans_slice;
  OptixBuildInput build_in;
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
};
struct SceneElement {
  // This is borrowed from the children scene objects so remember to keep them alive.
  const AsFeedback* asfb;
  Transform trans;
  DeviceMemorySlice mat_devmem;
};
// Traversable object collection.
struct Scene {
  // Acceleration structure build feedback.
  AsFeedback* inner;
  // Scene objects as components in the scene
  std::vector<SceneElement> elems;
};


struct Transaction {
  CUstream stream;
  std::vector<DeviceMemory> mnged_devmems;
  std::vector<void*> mnged_hostmems;
};




//
// Extensions.
//

namespace ext {

struct NaivePipelineConfig {
  bool debug;
  const char* ptx_data;
  size_t ptx_size;
  unsigned trace_depth;
  uint32_t max_ninst;
  const char* rg_name;
  const char* ms_name;
  const char* ah_name;
  const char* ch_name;
  // Size of Ray property material to be allocated. This material will be
  // accessible in the raygen stage of the pipeline, by the base address
  // returned from `optixGetSbtDataPointer`.
  size_t ray_prop_size;
  // Size of Environment material to be allocated. This material will be
  // accessible in the miss stage of the pipeline, by the base address returned
  // from `optixGetSbtDataPointer`.
  size_t env_size;
  // Size of Hit material to be allocated. This material will be accessible in
  // intersection, any-hit and closest-hit programs of the pipeline by the base
  // address returned from `optixGetSbtDataPointer`. This is per-instance data.
  size_t mat_size;
  // Size of Launch config to be allocated. This material will be accessible in
  // all stages of the pipeline, by the global constant variable defined with
  // `LAUNCH_CFG` attribute.
  size_t launch_cfg_size;
};



} // namespace ext


} // namespace liong
