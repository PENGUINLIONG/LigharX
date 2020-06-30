#include "core.hpp"
#include "log.hpp"

namespace {

void log_cb(liong::log::LogLevel lv, const std::string& msg) {
  using liong::log::LogLevel;
  switch (lv) {
  case LogLevel::L_LOG_LEVEL_INFO:
    printf("[\x1B[32mINFO\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_WARNING:
    printf("[\x1B[33mWARN\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_ERROR:
    printf("[\x1B[31mERROR\x1B[0m] %s\n", msg.c_str());
    break;
  }
}

}

using namespace liong;


PipelineConfig l_create_pipe_cfg() {
  // Note: For Visual Studio 2019 the default working directory is
  // `${CMAKE_BINARY_DIR}/bin`
  auto pipe_cfg = PipelineConfig {};
  pipe_cfg.debug = true;
  pipe_cfg.mod_path = "../assets/demo.ptx";
  pipe_cfg.launch_param_name = "cfg";
  pipe_cfg.npayload_wd = 2;
  pipe_cfg.nattr_wd = 2;
  pipe_cfg.trace_depth = 2;
  {
    auto rg_cfg = PipelineStageConfig { "__raygen__", nullptr };
    pipe_cfg.rg_cfg = rg_cfg;
  }
  {
    auto hitgrp_cfg = PipelineHitGroupConfig {};
    hitgrp_cfg.ch_name = "__closesthit__";
    hitgrp_cfg.ah_name = "__anyhit__";
    pipe_cfg.hitgrp_cfgs.push_back(hitgrp_cfg);
  }
  {
    auto ms_cfg = PipelineStageConfig { "__miss__", nullptr };
    pipe_cfg.ms_cfgs.push_back(ms_cfg);
  }
  return pipe_cfg;
}

MeshConfig l_create_mesh_cfg() {
  const static float vert_buf[] = {
    -0.5, 0.5, 0.0,
    0.5, 0.5, 0.0,
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0
  };
  const static uint16_t idx_buf[] = {
    2, 3, 1,
    2, 1, 0
  };

  auto mesh_cfg = MeshConfig {};
  mesh_cfg.vert_buf = vert_buf;
  mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;
  mesh_cfg.nvert = 4;
  mesh_cfg.vert_stride = 3 * sizeof(float);
  mesh_cfg.idx_buf = idx_buf;
  mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
  mesh_cfg.ntri = 2;
  mesh_cfg.tri_stride = 3 * sizeof(uint16_t);

  return mesh_cfg;
}

MeshConfig l_create_cube_cfg(const mat3x4& world2obj) {
  auto obj2world = world2obj.inverse();
  const float p = 0.5;
  const float n = -0.5;
  static const float verts[] = {
    n, p, n,
    n, p, p,
    p, p, p,
    p, p, n,
    n, n, n,
    n, n, p,
    p, n, p,
    p, n, n
  };
  const uint32_t a = 0;
  const uint32_t b = 1;
  const uint32_t c = 2;
  const uint32_t d = 3;
  const uint32_t e = 4;
  const uint32_t f = 5;
  const uint32_t g = 6;
  const uint32_t h = 7;
  static const uint16_t idxs[] = {
    f, e, a,   f, a, b,
    g, f, b,   g, b, c,
    h, g, c,   h, c, d,
    e, h, d,   e, d, a,
    a, d, c,   a, c, b,
    e, f, g,   e, g, h
  };
  MeshConfig mesh_cfg {};
  mesh_cfg.vert_buf = verts;
  mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;
  mesh_cfg.nvert = 8;
  mesh_cfg.vert_stride = 12;
  mesh_cfg.idx_buf = idxs;
  mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
  mesh_cfg.ntri = 2;
  mesh_cfg.tri_stride = 6;
  return mesh_cfg;
}

MeshConfig l_create_pln_cfg(const mat3x4& world2obj) {
  auto obj2world = world2obj.inverse();
  static const float verts[] = {
    -0.5, 0.0, -0.5,
    -0.5, 0.0, 0.5,
    0.5, 0.0, 0.5,
    0.5, 0.0, -0.5
  };
  static const uint16_t idxs[] = {
    0, 1, 2,   0, 2, 3,
  };
  MeshConfig mesh_cfg {};
  mesh_cfg.vert_buf = verts;
  mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;
  mesh_cfg.nvert = 4;
  mesh_cfg.vert_stride = 12;
  mesh_cfg.idx_buf = idxs;
  mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
  mesh_cfg.ntri = 1;
  mesh_cfg.tri_stride = 6;
  return mesh_cfg;
}




int main() {
  liong::log::set_log_callback(log_cb);

  //std::string xxx;
  //std::cin >> xxx;

  initialize();
  liong::log::info("optix lab started");


  auto pipe_cfg = l_create_pipe_cfg();
  auto mesh_cfg = l_create_mesh_cfg();

  Context ctxt;
  Pipeline pipe;
  Framebuffer framebuf;
  Mesh mesh;
  SceneObject sobj;
  Scene scene;
  Transaction transact;

  try {
    ctxt = create_ctxt();
    pipe = create_pipe(ctxt, pipe_cfg);
    framebuf = create_framebuf(32, 32);
    mesh = create_mesh(mesh_cfg);
    sobj = create_sobj();
    scene = create_scene({ sobj });
    transact = create_transact();

    cmd_build_sobj(transact, ctxt, mesh, sobj);
    wait_transact(transact);
    cmd_compact_mem(transact, ctxt, sobj);
    wait_transact(transact);
    cmd_build_scene(transact, ctxt, scene);
    wait_transact(transact);
    cmd_compact_mem(transact, ctxt, scene);
    wait_transact(transact);
    cmd_traverse(transact, pipe, framebuf, scene);
    wait_transact(transact);

    snapshot_framebuf(framebuf, "./snapshot.bmp");

    liong::log::info("sounds good");
  } catch (const std::exception& e) {
    liong::log::error("application threw an exception");
    liong::log::error(e.what());
  } catch (...) {
    liong::log::error("application threw an illiterate exception");
  }
  destroy_transact(transact);
  destroy_scene(scene);
  destroy_sobj(sobj);
  destroy_mesh(mesh);
  destroy_framebuf(framebuf);
  destroy_pipe(pipe);
  destroy_ctxt(ctxt);

  liong::log::info("optix lab ended");

  return 0;
}

