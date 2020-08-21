#pragma once
// Extensions to core functionalities.
// @PENGUINLIONG
#include <functional>
#ifdef L_USE_ASSIMP
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#endif // L_USE_ASSIMP
#include "core.hpp"

namespace liong {

namespace ext {

// Read PTX representation from a single file specified in `pipe_cfg`.
std::vector<char> read_ptx(const char* ptx_path);

// Create a naive pipeline that is enough to do simple ray tracing tasks. A
// naive pipline is a predefined cofiguration with `lauch_param_name` defaulted
// to `cfg`, `npayload_wd` and `nattr_wd` defaulted to 2 (to carry a pointer).
Pipeline create_naive_pipe(
  const Context& ctxt,
  const NaivePipelineConfig& naive_pipe_cfg
);

inline DeviceMemorySlice slice_naive_pipe_ray_prop(
  const Pipeline& pipe,
  const PipelineData& pipe_data
) {
  return slice_pipe_data(pipe, pipe_data, OPTIX_PROGRAM_GROUP_KIND_RAYGEN, 0);
}
inline DeviceMemorySlice slice_naive_pipe_mat(
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  uint32_t idx
) {
  return slice_pipe_data(pipe, pipe_data, OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    idx);
}
inline DeviceMemorySlice slice_naive_pipe_env(
  const Pipeline& pipe,
  const PipelineData& pipe_data
) {
  return slice_pipe_data(pipe, pipe_data, OPTIX_PROGRAM_GROUP_KIND_MISS, 0);
}



inline std::vector<Mesh> create_meshes(
  const std::vector<MeshConfig>& mesh_cfgs
) {
  std::vector<Mesh> rv;
  rv.resize(mesh_cfgs.size());
  for (const auto& mesh : mesh_cfgs) {
    create_mesh(mesh);
  }
  return rv;
}
inline void destroy_meshes(std::vector<Mesh>& meshes) {
  for (auto& mesh : meshes) {
    destroy_mesh(mesh);
  }
  meshes.clear();
}
std::vector<Mesh> import_meshes_from_file(const char* path);




inline std::vector<SceneObject> create_sobjs(size_t nsobj) {
  std::vector<SceneObject> rv;
  rv.resize(nsobj);
  for (auto i = 0; i < nsobj; ++i) {
    rv[i] = create_sobj();
  }
  return std::move(rv);
}
inline void destroy_sobjs(std::vector<SceneObject>& sobjs) {
  for (auto& sobj : sobjs) {
    destroy_sobj(sobj);
  }
  sobjs.clear();
}



inline void cmd_build_sobjs(
  Transaction& transact,
  const Context& ctxt,
  const std::vector<Mesh>& meshes,
  std::vector<SceneObject>& sobjs,
  bool can_compact = true
) {
  const auto n = meshes.size();
  ASSERT << (n == sobjs.size())
    << "mesh count and scene object count mismatched";
  for (auto i = 0; i < n; ++i) {
    cmd_build_sobj(transact, ctxt, meshes[i], sobjs[i], can_compact);
  }
}



struct SnapshotCommonHeader {
  // Snapshot magic number, MUST be set to `0x894C4A33` (`.LJ3` in big endian,
  // note that the fist character is not a char but a non-textual character).
  // The magic number can also be used to identify endian. Please ensure you
  // initialized the header with `init_snapshot_common_header`, so this field is
  // correctly filled.
  uint32_t magic = 0x894C4A33;
  // Common header version.
  uint32_t version = 0x0000001;
  // Type code of snapshot content. If a snapshot type has multiple versions,
  // version number or other semantical constructs should be encoded here.
  uint32_t type;
  // Optional size of data element in bytes. If `stride` is not zero, the import
  // procedure will enforce to consume at least and no more than this amount of
  // data; otherwise, it will exhaust the file and it will be the application's
  // work to infer the correct amount of data.
  uint32_t stride = 0;
  // Dimensions of homogeneous data element in `data`.
  uint4 dim = { 1, 1, 1, 1 };
};
// Take a snapshot of host memory content and writei it to a binary file. The
// binary data will follow a header section.
//
// WARNING: Be aware of endianess.
extern void snapshot_hostmem(
  const void* hostmem,
  size_t hostmem_size,
  const void* head,
  size_t head_size,
  const char* path
);
// Take a snapshot of device memory content and write it to a binary file. The
// binary data will follow a header section.
//
// WARNING: Be aware of endianess.
extern void snapshot_devmem(
  const DeviceMemorySlice& devmem,
  const void* head,
  size_t head_size,
  const char* path
);
// Buffer snapshots with a `SnapshotCommonHeader` header is a `CommonSnapshot`.
struct CommonSnapshot {
  // Dynamically allocated data buffer.
  void* data;
  // Type code of snapshot content.
  uint32_t type;
  // Size of data element in bytes. The import procedure will enforce to consume
  // at least and no more than this amount of data.
  uint32_t stride;
  // Number of homogeneous data element in `data`. Tensors should be filled in
  // column major order.
  uint4 dim;
  // Whether the buffer producer is an little endian machine.
  bool is_le;

  constexpr size_t nelem() const {
    return dim.x * dim.y * dim.z * dim.w;
  }
  constexpr size_t size() const {
    return nelem() * stride;
  }
};
// Import buffer snapshot from local storage.
extern CommonSnapshot import_common_snapshot(const char* path);
// Release the resources allocated for the imported snapshot.
extern void destroy_common_snapshot(CommonSnapshot& snapshot);

enum FramebufferSnapshotFormat {
  L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO,
  L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_BMP,
  L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_EXR,
};
// Take a snapshot of the framebuffer content and write it to a image file.
// Currently Lighar support 2 image file types. The output format is decided by
// `path` extension if `snapshot_fmt` is L_EXT_SNAPSHOT_FMT_AUTO.
//
// WARNING: This only works properly on little-endian platforms.
// WARNING: Snapshot format inferrence only works for lower-case extension
// names.
extern void snapshot_framebuf(
  const Framebuffer& framebuf,
  const char* path,
  FramebufferSnapshotFormat framebuf_snapshot_fmt
);
inline void snapshot_framebuf(
  const Framebuffer& framebuf,
  const char* path
) {
  snapshot_framebuf(framebuf, path, L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO);
}
extern void snapshot_host_framebuf(
  const void* framebuf,
  const uint32_t w,
  const uint32_t h,
  PixelFormat fmt,
  const char* path,
  FramebufferSnapshotFormat framebuf_snapshot_fmt
);
inline void snapshot_host_framebuf(
  const void* framebuf,
  const uint32_t w,
  const uint32_t h,
  PixelFormat fmt,
  const char* path
) {
  snapshot_host_framebuf(framebuf, w, h, fmt, path,
    L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO);
}

} // namespace ext

} // namespace liong
