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
Pipeline l_create_naive_pipe(
  const Context& ctxt,
  const std::vector<char>& ptx
) {
  // Note: For Visual Studio 2019 the default working directory is
  // `${CMAKE_BINARY_DIR}/bin`
  ext::NaivePipelineConfig naive_pipe_cfg {};
  naive_pipe_cfg.debug = true;
  naive_pipe_cfg.ptx_data = ptx.data();
  naive_pipe_cfg.ptx_size = ptx.size();
  naive_pipe_cfg.rg_name = "__raygen__";
  naive_pipe_cfg.ms_name = "__miss__";
  naive_pipe_cfg.ch_name = "__closesthit__";
  naive_pipe_cfg.ah_name = "__anyhit__";
  naive_pipe_cfg.env_size = sizeof(TEnv);
  naive_pipe_cfg.mat_size = sizeof(TMat);
  naive_pipe_cfg.trace_depth = TTravDepth;
  return ext::create_naive_pipe(ctxt, naive_pipe_cfg);
}

int main() {
  liong::log::set_log_callback(log_cb);

  //std::string xxx;
  //std::cin >> xxx;

  initialize();
  liong::log::info("optix lab started");

  liong::log::info(vec3{ 1.0, 1.0, 1.0 }.to_unorm_pack());

  Context ctxt;
  Pipeline pipe;
  PipelineData pipe_data;
  Framebuffer framebuf;
  std::vector<Mesh> meshes;
  std::vector<SceneObject> sobjs;
  Scene scene;
  Transaction transact;

  try {
    ctxt = create_ctxt();
    {
      auto ptx = ext::read_ptx("../assets/cuda_compile_ptx_1_generated_demo.cu.ptx");
      pipe = l_create_naive_pipe<Material, Environment, 5>(ctxt, ptx);
    }
    framebuf = create_framebuf(32, 32);
    meshes = ext::import_meshes_from_file("./untitled.obj");
    sobjs = ext::create_sobjs(meshes.size());
    scene = create_scene(sobjs);
    pipe_data = create_pipe_data(pipe);
    transact = create_transact();
    
    ext::cmd_build_sobjs(transact, ctxt, meshes, sobjs);
    wait_transact(transact);

    cmd_compact_mems(transact, ctxt, sobjs);
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
  ext::destroy_sobjs(sobjs);
  ext::destroy_meshes(meshes);
  destroy_framebuf(framebuf);
  destroy_pipe(pipe);
  destroy_ctxt(ctxt);

  liong::log::info("optix lab ended");

  return 0;
}
