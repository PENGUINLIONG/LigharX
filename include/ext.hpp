#pragma once
// Extensions to core functionalities.
// @PENGUINLIONG
#include <core.hpp>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

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


}

}
