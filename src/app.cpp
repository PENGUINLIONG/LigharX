#include "core.hpp"
#include "ext.hpp"
#include "log.hpp"
#include "x-mat.hpp"

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

// Create pipeline config with scene object material (TMat) and environment
// material (TEnv) which is used during ray miss.
template<typename TMat, typename TEnv, size_t TTravDepth>
Pipeline l_create_naive_pipe(const Context& ctxt) {
  // Note: For Visual Studio 2019 the default working directory is
  // `${CMAKE_BINARY_DIR}/bin`
  ext::NaivePipelineConfig naive_pipe_cfg {};
  naive_pipe_cfg.debug = true;
  naive_pipe_cfg.mod_path = "../assets/demo.ptx";
  naive_pipe_cfg.rg_name = "__raygen__";
  naive_pipe_cfg.ms_name = "__miss__";
  naive_pipe_cfg.ch_name = "__closesthit__";
  naive_pipe_cfg.ah_name = "__anyhit__";
  naive_pipe_cfg.env_size = sizeof(TEnv);
  naive_pipe_cfg.mat_size = sizeof(TMat);
  naive_pipe_cfg.trace_depth = TTravDepth;
  return ext::create_native_pipe(ctxt, naive_pipe_cfg);
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

  liong::log::info(vec3{ 1.0, 1.0, 1.0 }.to_unorm_pack());



  auto mesh_cfg = l_create_mesh_cfg();

  Context ctxt;
  Pipeline pipe;
  PipelineData pipe_data;
  Framebuffer framebuf;
  Mesh mesh;
  SceneObject sobj;
  Scene scene;
  Transaction transact;

  try {
    ctxt = create_ctxt();
    pipe = l_create_naive_pipe<Material, Environment, 5>(ctxt);
    framebuf = create_framebuf(32, 32);
    mesh = create_mesh(mesh_cfg, 0);
    sobj = create_sobj();
    scene = create_scene({ sobj });
    pipe_data = create_pipe_data(pipe);
    transact = create_transact();

    cmd_build_sobj(transact, ctxt, mesh, sobj);
    wait_transact(transact);

    cmd_compact_mem(transact, ctxt, sobj);
    wait_transact(transact);

    cmd_build_scene(transact, ctxt, scene);
    wait_transact(transact);

    cmd_compact_mem(transact, ctxt, scene);
    wait_transact(transact);

    cmd_init_pipe_data(transact, pipe, pipe_data);
    vec3 sky_color[1] = { vec3 { 0.5, 0.5, 0.5 } };
    auto env_slice = ext::slice_naive_pipe_env(pipe, pipe_data);
    cmd_upload_mem(transact, sky_color, env_slice, sizeof(vec3));
    vec3 obj_color[1] = { vec3 { 1.0, 0.0, 1.0 } };
    auto mat_slice = ext::slice_naive_pipe_mat(pipe, pipe_data, 0);
    cmd_upload_mem(transact, obj_color, mat_slice, sizeof(vec3));
    wait_transact(transact);

    cmd_traverse(transact, pipe, pipe_data, framebuf, scene);
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
  destroy_pipe_data(pipe_data);
  destroy_scene(scene);
  destroy_sobj(sobj);
  destroy_mesh(mesh);
  destroy_framebuf(framebuf);
  destroy_pipe(pipe);
  destroy_ctxt(ctxt);

  liong::log::info("optix lab ended");

  return 0;
}
