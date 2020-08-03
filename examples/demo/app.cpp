#include "core.hpp"
#include "ext.hpp"
#include "log.hpp"
#include "mat.hpp"
#include "denoise.hpp"

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
template<size_t TTravDepth>
Pipeline l_create_naive_pipe(
  const Context& ctxt,
  const std::vector<char>& ptx,
  const mat::MaterialType& ray_prop,
  const mat::MaterialType& env,
  const mat::MaterialType& mat,
  const mat::MaterialType& launch_cfg
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
  naive_pipe_cfg.ray_prop_size = ray_prop.size;
  naive_pipe_cfg.env_size = env.size;
  naive_pipe_cfg.mat_size = mat.size;
  naive_pipe_cfg.trace_depth = TTravDepth;
  naive_pipe_cfg.max_ninst = 3;
  naive_pipe_cfg.launch_cfg_size = launch_cfg.size;
  return ext::create_naive_pipe(ctxt, naive_pipe_cfg);
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


// FIXME: (penguinliong) This cannot be rendered.
MeshConfig l_create_pln_cfg(const Transform& world2obj) {
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
  mesh_cfg.pretrans = world2obj;
  mesh_cfg.vert_buf = verts;
  mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;
  mesh_cfg.nvert = 4;
  mesh_cfg.vert_stride = 12;
  mesh_cfg.idx_buf = idxs;
  mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
  mesh_cfg.ntri = 2;
  mesh_cfg.tri_stride = 6;
  return mesh_cfg;
}

MeshConfig l_create_cube_cfg(const Transform& world2obj) {
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
  mesh_cfg.pretrans = world2obj;
  mesh_cfg.vert_buf = verts;
  mesh_cfg.vert_fmt = OPTIX_VERTEX_FORMAT_FLOAT3;
  mesh_cfg.nvert = 8;
  mesh_cfg.vert_stride = 12;
  mesh_cfg.idx_buf = idxs;
  mesh_cfg.idx_fmt = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
  mesh_cfg.ntri = 12;
  mesh_cfg.tri_stride = 6;
  return mesh_cfg;
}

int main() {
  liong::log::set_log_callback(log_cb);

  //std::string xxx;
  //std::cin >> xxx;

  initialize();
  liong::log::info("optix lab started");

  Context ctxt;
  Pipeline pipe;
  PipelineData pipe_data;
  Framebuffer framebuf;
  std::vector<Mesh> meshes;
  std::vector<SceneObject> sobjs;
  Scene scene;
  Transaction transact;
  denoise::Denoiser denoiser;

  // Define materials.
  mat::MaterialType ray_prop_ty {};
  mat::MaterialType env_ty {};
  mat::push_mat_ty_entry(env_ty, "ambient", sizeof(float3));
  mat::MaterialType mat_ty {};
  mat::push_mat_ty_entry(mat_ty, "albedo", sizeof(float3));
  mat::push_mat_ty_entry(mat_ty, "emit", sizeof(float3));
  mat::MaterialType launch_cfg_ty {};
  mat::push_mat_ty_entry(launch_cfg_ty, "trav", sizeof(OptixTraversableHandle));
  mat::push_mat_ty_entry(launch_cfg_ty, "framebuf", sizeof(CUdeviceptr));

  mat::Material env;
  mat::Material mat;
  mat::Material launch_cfg;
  
  try {
    ctxt = create_ctxt();
    {
      auto ptx = ext::read_ptx("../assets/cuda_compile_ptx_1_generated_demo.cu.ptx");
      pipe = l_create_naive_pipe<2>(ctxt, ptx, ray_prop_ty, env_ty, mat_ty,
        launch_cfg_ty);
    }
    framebuf = create_framebuf(L_FORMAT_R8G8B8A8_UNORM, { 256, 256, 1 });
    //meshes = ext::import_meshes_from_file("./untitled.obj");
    // Initialize meshes.
    meshes = {};
    meshes.push_back(create_mesh(l_create_cube_cfg(Transform {})));
    sobjs = ext::create_sobjs(meshes.size());
    scene = create_scene();


    /* Cube 1 */ {
      auto trans = Transform()
        .translate(-1, 0.75, 0);
      add_scene_sobj(scene, sobjs[0], trans, DeviceMemorySlice {});
    }
    /* Cube 2 */ {
      auto trans = Transform()
        .translate(-0.75, 0, 0)
        .rotate(normalize(make_float3(1, 1, 0)), deg2rad(-15))
        .translate(0, 0.75, 1.75);
      add_scene_sobj(scene, sobjs[0], trans, DeviceMemorySlice {});
    }
    /* Cube 3 */ {
      auto trans = Transform()
        .translate(-1, -0.75, 0);
      add_scene_sobj(scene, sobjs[0], trans, DeviceMemorySlice {});
    }
    



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
    wait_transact(transact);

    /* Environment material */ {
      env = mat::create_mat(env_ty);
      const float3 ambient = unquant_unorm8_rgb(250, 250, 250);
      mat::assign_mat_entry(env_ty, env, "ambient", &ambient, sizeof(ambient));
      auto env_slice = ext::slice_naive_pipe_env(pipe, pipe_data);
      upload_mem(env.data, env_slice, env.size);
      mat::destroy_mat(env);
    }
    /* Cube 1 material */ {
      mat = mat::create_mat(mat_ty);
      const float3 albedo = unquant_unorm8_rgb(245, 228, 0);
      const float3 emit = unquant_unorm8_rgb(245, 228, 0);
      mat::assign_mat_entry(mat_ty, mat, "albedo", &albedo, sizeof(albedo));
      mat::assign_mat_entry(mat_ty, mat, "emit", &emit, sizeof(emit));
      auto mat_slice = ext::slice_naive_pipe_mat(pipe, pipe_data, 0);
      upload_mem(mat.data, mat_slice, mat.size);
      mat::destroy_mat(mat);
    }
    /* Cube 2 material */ {
      mat = mat::create_mat(mat_ty);
      const float3 albedo = unquant_unorm8_rgb(68, 228, 235);
      const float3 emit = unquant_unorm8_rgb(32, 173, 150);
      mat::assign_mat_entry(mat_ty, mat, "albedo", &albedo, sizeof(albedo));
      mat::assign_mat_entry(mat_ty, mat, "emit", &emit, sizeof(emit));
      auto mat_slice = ext::slice_naive_pipe_mat(pipe, pipe_data, 1);
      upload_mem(mat.data, mat_slice, mat.size);
      mat::destroy_mat(mat);
    }
    /* Cube 3 material */ {
      mat = mat::create_mat(mat_ty);
      const float3 albedo = unquant_unorm8_rgb(230, 50, 70);
      const float3 emit = unquant_unorm8_rgb(150, 32, 55);
      mat::assign_mat_entry(mat_ty, mat, "albedo", &albedo, sizeof(albedo));
      mat::assign_mat_entry(mat_ty, mat, "emit", &emit, sizeof(emit));
      auto mat_slice = ext::slice_naive_pipe_mat(pipe, pipe_data, 2);
      upload_mem(mat.data, mat_slice, mat.size);
      mat::destroy_mat(mat);
    }
    /* Launch config */ {
      launch_cfg = mat::create_mat(launch_cfg_ty);
      const OptixTraversableHandle trav = scene.inner->trav;
      const CUdeviceptr fb = framebuf.framebuf_devmem.ptr;
      mat::assign_mat_entry(launch_cfg_ty, launch_cfg,
        "trav", &trav, sizeof(trav));
      mat::assign_mat_entry(launch_cfg_ty, launch_cfg,
        "framebuf", &fb, sizeof(fb));
      auto launch_cfg_slice = slice_pipe_launch_cfg(pipe, pipe_data);
      upload_mem(launch_cfg.data, launch_cfg_slice, launch_cfg.size);
      mat::destroy_mat(launch_cfg);
    }

    cmd_traverse(transact, pipe, pipe_data, framebuf.dim);
    wait_transact(transact);

    {
      denoise::DenoiserConfig denoiser_cfg {};
      denoiser_cfg.fmt = framebuf.fmt;
      denoiser_cfg.hdr_intensity = 0;
      denoiser_cfg.in_kind = OPTIX_DENOISER_INPUT_RGB;
      denoiser_cfg.max_dim = framebuf.dim;
      denoiser_cfg.inplace = true;
      denoiser = denoise::create_denoiser(ctxt, denoiser_cfg);
    }
    denoise::cmd_denoise(transact, denoiser, framebuf, framebuf);
    wait_transact(transact);

    ext::snapshot_framebuf(framebuf, "./snapshot.bmp");

    liong::log::info("sounds good");
  } catch (const std::exception& e) {
    liong::log::error("application threw an exception");
    liong::log::error(e.what());
    liong::log::error("application cannot continue");
  } catch (...) {
    liong::log::error("application threw an illiterate exception");
  }
  denoise::destroy_denoiser(denoiser);
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
