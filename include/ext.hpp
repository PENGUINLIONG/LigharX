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



// Take a snapshot of the device memory content and write it to a Binary file.
// The binary data will follow a header section.
//
// WARNING: Be aware of endianess.
extern void snapshot_devmem(
  const DeviceMemorySlice& devmem,
  const void* head,
  size_t head_size,
  const char* path
);
// Take a snapshot of the device memory content and write it to a Binary file.
//
// WARNING: Be aware of endianess.
extern void snapshot_devmem(const DeviceMemorySlice& devmem, const char* path);
// Take a snapshot of the framebuffer content and write it to a BMP file.
//
// WARNING: This only works properly on little-endian platforms.
extern void snapshot_framebuf(const Framebuffer& framebuf, const char* path);

} // namespace ext

} // namespace liong
