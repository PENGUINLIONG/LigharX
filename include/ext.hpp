#pragma once
// Extensions to core functionalities.
// @PENGUINLIONG
#include <core.hpp>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

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



inline std::vector<Mesh> create_meshes(
  const std::vector<MeshConfig>& mesh_cfgs,
  size_t mat_size = 0
) {
  std::vector<Mesh> rv;
  rv.resize(mesh_cfgs.size());
  for (const auto& mesh : mesh_cfgs) {
    create_mesh(mesh, mat_size);
  }
  return rv;
}
inline void destroy_meshes(std::vector<Mesh>& meshes) {
  for (auto& mesh : meshes) {
    destroy_mesh(mesh);
  }
  meshes.clear();
}
std::vector<Mesh> import_meshes_from_file(const char* path) {
  const aiScene* raw_scene = aiImportFile(
    path,
    aiProcess_Triangulate |
    aiProcess_JoinIdenticalVertices |
    aiProcess_SortByPType
  );
  ASSERT << raw_scene
    << "cannot import scene";
  ASSERT << (raw_scene->mNumMeshes > 0)
    << "imported scene has no mesh";

  std::vector<Mesh> mesh;
  mesh.reserve(raw_scene->mNumMeshes);

  for (auto i = 0; i < raw_scene->mNumMeshes; ++i) {
    const auto& raw_mesh = *raw_scene->mMeshes[i];
    MeshConfig mesh_cfg {};

    mesh_cfg.vert_buf = raw_mesh.mVertices;
    mesh_cfg.nvert = raw_mesh.mNumVertices;
    mesh_cfg.vert_stride =
      sizeof(std::remove_pointer_t<decltype(raw_mesh.mVertices)>);
    mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;

    std::vector<uint32_t> idx_buf;
    idx_buf.resize(raw_mesh.mNumFaces * 3); // All faces has been triangulated.
    for (auto j = 0; j < raw_mesh.mNumFaces; ++j) {
      const auto& face = raw_mesh.mFaces[j];
      for (auto k = 0; k < face.mNumIndices; ++k) {
        idx_buf[j * 3 + k] = face.mIndices[k];
      }
    }
    mesh_cfg.idx_buf = idx_buf.data();
    mesh_cfg.ntri = raw_mesh.mNumFaces;
    mesh_cfg.tri_stride = 3 * sizeof(uint32_t);
    mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    // DO NOT REPLACE WITH `create_meshes`; NOTICE THE LIFETIME OF `idx_buf`!
    mesh.emplace_back(create_mesh(mesh_cfg));
  }
  aiReleaseImport(raw_scene);
  return mesh;
}




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





}

}
