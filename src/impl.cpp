#define NOMINMAX
#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include "ty.hpp"
#include "core.hpp"
#include "ext.hpp"
#include "log.hpp"
#include "except.hpp"
// !!! MUST ONLY DEFINE ONCE !!!
#include <optix_function_table_definition.h>

namespace liong {

//
// Core functionalities.
//

void initialize() {
  CUDA_ASSERT << cuInit(0);
  OPTIX_ASSERT << optixInit();
}


void _optix_log_cb(unsigned level, const char* tag, const char* msg, void* _) {
  std::stringstream buf {};
  auto beg = msg;
  auto pos = beg;
  char c;
  do {
    if (*pos == '\n') {
      buf.str("");
      buf << tag << " - " << std::string(beg, pos);
      switch (level) {
      case 1: liong::log::error(buf.str()); break;
      case 2: liong::log::error(buf.str()); break;
      case 3: liong::log::warn(buf.str()); break;
      case 4: liong::log::info(buf.str()); break;
      }
      beg = ++pos;
    } else {
      ++pos;
    }
  } while (*pos != '\0');
}
Context create_ctxt(int dev_idx) {
  CUdevice dev;
  CUDA_ASSERT << cuDeviceGet(&dev, dev_idx);

  CUcontext cuda_ctxt;
  CUDA_ASSERT << cuCtxCreate(&cuda_ctxt, CU_CTX_MAP_HOST, dev);

  OptixDeviceContext optix_dc;
  auto opts = OptixDeviceContextOptions { _optix_log_cb, nullptr, 4 };
  OPTIX_ASSERT << optixDeviceContextCreate(cuda_ctxt, &opts, &optix_dc);

  liong::log::info("created optix context");
  return Context { dev, cuda_ctxt, optix_dc };
}
void destroy_ctxt(Context& ctxt) {
  if (ctxt.optix_dc) {
    OPTIX_ASSERT << optixDeviceContextDestroy(ctxt.optix_dc);
  }
  ctxt = {};
  liong::log::info("destroyed optix context");
}


DeviceMemory alloc_mem(size_t size, size_t align) {
  if (size == 0) { return DeviceMemory {}; }
  CUdeviceptr devmem {};
  auto aligned_size = align_size(size, align);
  CUDA_ASSERT << cuMemAlloc(&devmem, aligned_size);
  liong::log::info("allocated memory of ", size, " bytes");
  return DeviceMemory {
    devmem,
    align_addr(devmem, align),
    aligned_size,
    size
  };
}
void free_mem(DeviceMemory& devmem) {
  if (devmem.alloc_base) { CUDA_ASSERT << cuMemFree(devmem.alloc_base); }
  liong::log::info("freed memory of ", devmem.size, " bytes");
  devmem = {};
}

void transfer_mem(const DeviceMemorySlice& src, const DeviceMemorySlice& dst) {
  ASSERT << (src.size <= dst.size)
    << "transfer out of range";
  ASSERT << (((src.ptr < dst.ptr) && (src.ptr + src.size <= dst.ptr)) ||
    ((src.ptr > dst.ptr) && (src.ptr >= dst.ptr + dst.size)))
    << "transfer range overlapped";
  CUDA_ASSERT << cuMemcpy(dst.ptr, src.ptr, src.size);
}
void upload_mem(const void* src, const DeviceMemorySlice& dst, size_t size) {
  if (size == 0) { return; }
  ASSERT << (size <= dst.size)
    << "memory write out of range";
  CUDA_ASSERT << cuMemcpyHtoD(dst.ptr, src, size);
}
void download_mem(const DeviceMemorySlice& src, void* dst, size_t size) {
  if (size == 0) { return; }
  CUDA_ASSERT << cuMemcpyDtoH(dst, src.ptr, size);
}

DeviceMemory shadow_mem(const void* buf, size_t size, size_t align) {
  auto devmem = alloc_mem(size, align);
  upload_mem(buf, devmem, size);
  return devmem;
}

Texture create_tex(
  const Context& ctxt,
  const TextureConfig& tex_cfg,
  const SamplerConfig& sampler_cfg
) {
  CUtexObject tex;
  CUDA_RESOURCE_DESC rsc_desc {};
  rsc_desc.resType = CU_RESOURCE_TYPE_PITCH2D;
  rsc_desc.res.pitch2D.devPtr = tex_cfg.tex_slice.ptr;
  rsc_desc.res.pitch2D.format = tex_cfg.fmt.get_cuda_fmt();
  rsc_desc.res.pitch2D.numChannels = tex_cfg.fmt.get_ncomp();
  rsc_desc.res.pitch2D.width = tex_cfg.w;
  rsc_desc.res.pitch2D.height = tex_cfg.h;
  rsc_desc.res.pitch2D.pitchInBytes = tex_cfg.row_align;
  CUDA_TEXTURE_DESC tex_desc {};
  tex_desc.addressMode[0] = sampler_cfg.addr_mode;
  tex_desc.addressMode[1] = sampler_cfg.addr_mode;
  tex_desc.addressMode[2] = sampler_cfg.addr_mode;
  tex_desc.filterMode = sampler_cfg.filter_mode;
  CUDA_RESOURCE_VIEW_DESC rsc_view_desc {};
  // Use what we have in `rsc_desc`.
  rsc_view_desc.format = CU_RES_VIEW_FORMAT_NONE;
  rsc_view_desc.width = tex_cfg.w;
  rsc_view_desc.height = tex_cfg.h;
  rsc_view_desc.depth = tex_cfg.d;
  CUDA_ASSERT << cuTexObjectCreate(&tex, &rsc_desc, &tex_desc, &rsc_view_desc);
  return Texture { tex, tex_cfg };
}
void destroy_tex(Texture& tex) {
  CUDA_ASSERT << cuTexObjectDestroy(tex.tex);
  tex.tex = {};
  tex.tex_cfg = {};
}


OptixModule _create_mod(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg
) {
  const size_t LOG_LEN = 400;
  char log[LOG_LEN];
  size_t log_len = LOG_LEN;

  // Create optix module from PTX.
  OptixModuleCompileOptions mod_opt {
    0,
    pipe_cfg.debug ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 :
      OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
    pipe_cfg.debug ? OPTIX_COMPILE_DEBUG_LEVEL_FULL :
      OPTIX_COMPILE_DEBUG_LEVEL_NONE,
  };
  OptixPipelineCompileOptions pipe_opt {
    0,
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
    pipe_cfg.npayload_wd,
    pipe_cfg.nattr_wd,
    OPTIX_EXCEPTION_FLAG_NONE,
    pipe_cfg.launch_cfg_name,
  };
  OptixModule mod;
  auto res = optixModuleCreateFromPTX(ctxt.optix_dc, &mod_opt, &pipe_opt,
    (const char*)pipe_cfg.ptx_data, pipe_cfg.ptx_size, log, &log_len, &mod);
  if (log_len != 0 && res != OPTIX_SUCCESS) {
    liong::log::warn(log);
  }
  OPTIX_ASSERT << res;
  return mod;
}
struct PipelinePrep {
  std::vector<OptixProgramGroup> pgrps;
  PipelineLayout pipe_layout;
};
constexpr size_t _sbt_align(size_t size) {
  return align_addr(
    OPTIX_SBT_RECORD_HEADER_SIZE + size,
    OPTIX_SBT_RECORD_ALIGNMENT
  );
}
PipelinePrep _create_pipe_prep(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg,
  OptixModule mod
) {
  const size_t LOG_LEN = 400;
  char log[LOG_LEN];
  size_t log_len = LOG_LEN;

  PipelinePrep pipe_prep {};
  auto& pipe_layout = pipe_prep.pipe_layout;
  std::vector<OptixProgramGroupDesc> pgrp_descs {};
  // Conservative size of device memory to contain all data referred by SBT.
  size_t sbt_size = 0, sbt_kind_max_size;
  // ORDER IS IMPORTANT; DO NOT RESORT >>>
  pipe_layout.sbt_raygen_offset = sbt_size;
  if (pipe_cfg.rg_cfg.name) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = pipe_cfg.rg_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_layout.sbt_raygen_stride = _sbt_align(pipe_cfg.rg_cfg.data_size);
  }
  sbt_size += pipe_layout.sbt_raygen_stride;

  pipe_layout.sbt_except_offset = sbt_size;
  if (pipe_cfg.ex_cfg.name) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = pipe_cfg.ex_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_layout.sbt_except_stride = _sbt_align(pipe_cfg.ex_cfg.data_size);
  }
  sbt_size += pipe_layout.sbt_except_stride;

  pipe_layout.sbt_miss_offset = sbt_size;
  pipe_layout.nsbt_miss = pipe_cfg.ms_cfgs.size();
  for (auto& ms_cfg : pipe_cfg.ms_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = ms_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_layout.sbt_miss_stride =
      std::max(_sbt_align(ms_cfg.data_size), pipe_layout.sbt_miss_stride);
  }
  sbt_size += pipe_layout.sbt_miss_stride * pipe_layout.nsbt_miss;

  pipe_layout.sbt_hitgrp_offset = sbt_size;
  // TODO: (penguinliong) Check out actually how to implement multiple ray
  // types. Feels like it's not allowed to have multiple hit-groups in a single
  // pipeline?
  pipe_layout.nsbt_hitgrp = pipe_cfg.hitgrp_cfgs.size();
  for (auto& hitgrp_cfg : pipe_cfg.hitgrp_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (hitgrp_cfg.ah_name) {
      pgrp_desc.hitgroup.moduleAH = mod;
      pgrp_desc.hitgroup.entryFunctionNameAH = hitgrp_cfg.ah_name;
    }
    if (hitgrp_cfg.ch_name) {
      pgrp_desc.hitgroup.moduleCH = mod;
      pgrp_desc.hitgroup.entryFunctionNameCH = hitgrp_cfg.ch_name;
    }
    if (hitgrp_cfg.is_name) {
      pgrp_desc.hitgroup.moduleIS = mod;
      pgrp_desc.hitgroup.entryFunctionNameIS = hitgrp_cfg.is_name;
    }
    pgrp_descs.push_back(pgrp_desc);
    pipe_layout.sbt_hitgrp_stride =
      std::max(_sbt_align(hitgrp_cfg.data_size), pipe_layout.sbt_hitgrp_stride);
  }
  sbt_size += pipe_cfg.max_ninst * // Do note this.
    (pipe_layout.sbt_hitgrp_stride * pipe_layout.nsbt_hitgrp);

  pipe_layout.sbt_call_offset = sbt_size;
  pipe_layout.nsbt_call = pipe_cfg.call_cfgs.size();
  for (auto& call_cfg : pipe_cfg.call_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    if (call_cfg.is_cc) {
      pgrp_desc.callables.moduleCC = mod;
      pgrp_desc.callables.entryFunctionNameCC = call_cfg.name;
    }
    if (call_cfg.is_dc) {
      pgrp_desc.callables.moduleDC = mod;
      pgrp_desc.callables.entryFunctionNameDC = call_cfg.name;
    }
    pgrp_descs.push_back(pgrp_desc);
    pipe_layout.sbt_call_stride =
      std::max(_sbt_align(call_cfg.data_size), pipe_layout.sbt_call_stride);
  }
  sbt_size += pipe_layout.sbt_call_stride * pipe_layout.nsbt_call;
  // <<< ORDER IS IMPORTANT; DO NOT RESORT

  pipe_layout.launch_cfg_size = pipe_cfg.launch_cfg_size;

  OptixProgramGroupOptions opt {};
  auto& pgrps = pipe_prep.pgrps;
  pgrps.resize(pgrp_descs.size());
  auto res = optixProgramGroupCreate(ctxt.optix_dc, pgrp_descs.data(),
    pgrp_descs.size(), &opt, log, &log_len,
    const_cast<OptixProgramGroup*>(pgrps.data()));
  if (log_len != 0 && res != OPTIX_SUCCESS) {
    liong::log::warn(log);
  }
  OPTIX_ASSERT << res;

  pipe_layout.sbt_size = sbt_size;
  pipe_layout.max_ninst = pipe_cfg.max_ninst;
  return pipe_prep;
}
OptixPipeline _create_pipe(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg,
  const std::vector<OptixProgramGroup>& pgrps
) {
  const size_t LOG_LEN = 400;
  char log[LOG_LEN];
  size_t log_len = LOG_LEN;

  OptixPipelineCompileOptions pipe_opt {
    0,
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
    pipe_cfg.npayload_wd,
    pipe_cfg.nattr_wd,
    OPTIX_EXCEPTION_FLAG_NONE,
    pipe_cfg.launch_cfg_name,
  };
  OptixPipelineLinkOptions link_opt {
    pipe_cfg.trace_depth,
    OPTIX_COMPILE_DEBUG_LEVEL_FULL
  };
  OptixPipeline pipe;
  auto res = optixPipelineCreate(ctxt.optix_dc, &pipe_opt, &link_opt,
    pgrps.data(), pgrps.size(), log, &log_len, &pipe);
  if (log_len != 0 && res != OPTIX_SUCCESS) {
    liong::log::warn(log);
  }
  OPTIX_ASSERT << res;
  return pipe;
}


Pipeline create_pipe(const Context& ctxt, const PipelineConfig& pipe_cfg) {
  ASSERT << (pipe_cfg.max_ninst > 0)
    << "maximum instance count must be specified";
  auto mod = _create_mod(ctxt, pipe_cfg);
  auto pipe_prep = _create_pipe_prep(ctxt, pipe_cfg, mod);
  auto pipe = _create_pipe(ctxt, pipe_cfg, pipe_prep.pgrps);
  // Not necessarily setting the stack size because it will be computed
  // internally automatically.
  std::stringstream ss;
  liong::log::info("created pipeline from module");
  return Pipeline {
    mod,
    std::move(pipe_prep.pgrps),
    pipe,
    std::move(pipe_prep.pipe_layout)
  };
}
void destroy_pipe(Pipeline& pipe) {
  if (pipe.pipe) { OPTIX_ASSERT << optixPipelineDestroy(pipe.pipe); }
  for (auto pgrp : pipe.pgrps) {
    if (pgrp) { OPTIX_ASSERT << optixProgramGroupDestroy(pgrp); }
  }
  if (pipe.mod) { OPTIX_ASSERT << optixModuleDestroy(pipe.mod); }
  pipe = {};
  liong::log::info("destroyed pipeline");
}

PipelineData create_pipe_data(const Pipeline& pipe) {
  const auto& pipe_layout = pipe.pipe_layout;
  auto i = 0;
  OptixShaderBindingTable sbt {};
  size_t alloc_size = pipe.pipe_layout.sbt_size +
    pipe.pipe_layout.launch_cfg_size;
  DeviceMemory devmem = alloc_mem(alloc_size, OPTIX_SBT_RECORD_ALIGNMENT);
  DeviceMemorySlice sbt_devmem = devmem.slice(0, pipe.pipe_layout.sbt_size);
  DeviceMemorySlice launch_cfg_devmem = devmem.slice(pipe.pipe_layout.sbt_size);
  const auto base = sbt_devmem.ptr;

  // Just don't waste time guessing what does 'prep' means. This function
  // originally return a structure called `PipelinePrep`, i.e., some prepared
  // data that will be passed to another function during pipeline creation. Not
  // changed hare because changing `prep_name` to `layout_name` will make some
  // of the following lines too long to be stuffed in width of 80.
  //
  // I can use linebreaks, I know. Why I have been so resistant to create new
  // lines while I'm writing this crazy shing up.
#define L_FILL_SBT_SINGLE_ENTRY(optix_name, prep_name, cfg_name)               \
  if (pipe_layout.sbt_##prep_name##_stride) {                                  \
    sbt.optix_name##Record = base + pipe_layout.sbt_##prep_name##_offset;      \
    ++i;                                                                       \
  }
#define L_FILL_SBT_MULTI_ENTRY(optix_name, prep_name)                          \
  if (pipe_layout.nsbt_##prep_name) {                                          \
    sbt.optix_name##RecordBase = base + pipe_layout.sbt_##prep_name##_offset;  \
    sbt.optix_name##RecordStrideInBytes = pipe_layout.sbt_##prep_name##_stride;\
    sbt.optix_name##RecordCount = pipe_layout.nsbt_##prep_name;                \
  }

  // ORDER IS IMPORTANT; DO NOT RESORT >>>
  L_FILL_SBT_SINGLE_ENTRY(raygen, raygen, rg);
  L_FILL_SBT_SINGLE_ENTRY(exception, except, ex);
  L_FILL_SBT_MULTI_ENTRY(miss, miss);
  L_FILL_SBT_MULTI_ENTRY(hitgroup, hitgrp);
  L_FILL_SBT_MULTI_ENTRY(callables, call);
  // <<< ORDER IS IMPORTANT; DO NOT RESORT

  liong::log::info("created pipeline data");
  return PipelineData {
    std::move(sbt),
    std::move(devmem),
    std::move(sbt_devmem),
    std::move(launch_cfg_devmem),
  };
}
void destroy_pipe_data(PipelineData& pipe_data) {
  free_mem(pipe_data.devmem);
  pipe_data = {};
  liong::log::info("destroyed pipeline data");
}
DeviceMemorySlice slice_pipe_data(
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  OptixProgramGroupKind kind,
  uint32_t idx
) {
  const auto& pipe_layout = pipe.pipe_layout;
  switch (kind) {
  case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
    return pipe_data.sbt_devmem.slice(
      pipe_layout.sbt_raygen_offset + OPTIX_SBT_RECORD_HEADER_SIZE,
      pipe_layout.sbt_raygen_stride - OPTIX_SBT_RECORD_HEADER_SIZE
    );
  case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
    return pipe_data.sbt_devmem.slice(
      pipe_layout.sbt_except_offset + OPTIX_SBT_RECORD_HEADER_SIZE,
      pipe_layout.sbt_except_stride - OPTIX_SBT_RECORD_HEADER_SIZE
    );
  case OPTIX_PROGRAM_GROUP_KIND_MISS:
    return pipe_data.sbt_devmem.slice(
      pipe_layout.sbt_miss_offset + pipe_layout.sbt_miss_stride * idx +
        OPTIX_SBT_RECORD_HEADER_SIZE,
      pipe_layout.sbt_miss_stride - OPTIX_SBT_RECORD_HEADER_SIZE
    );
  case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
    return pipe_data.sbt_devmem.slice(
      pipe_layout.sbt_hitgrp_offset + pipe_layout.sbt_hitgrp_stride * idx +
        OPTIX_SBT_RECORD_HEADER_SIZE,
      pipe_layout.sbt_hitgrp_stride - OPTIX_SBT_RECORD_HEADER_SIZE
    );
  case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
    return pipe_data.sbt_devmem.slice(
      pipe_layout.sbt_call_offset + pipe_layout.sbt_call_stride * idx +
        OPTIX_SBT_RECORD_HEADER_SIZE,
      pipe_layout.sbt_call_stride - OPTIX_SBT_RECORD_HEADER_SIZE
    );
  }
}
DeviceMemorySlice slice_pipe_launch_cfg(
  const Pipeline& pipe,
  const PipelineData& pipe_data
) {
  // TODO: (penguinliong)
  return pipe_data.launch_cfg_devmem;
}



Framebuffer create_framebuf(
  PixelFormat fmt,
  uint3 dim
) {
  ASSERT << ((dim.x != 0) && (dim.y != 0) && (dim.z != 0))
    << "framebuffer size cannot be zero";
  auto framebuf_devmem = alloc_mem(fmt.get_fmt_size() * dim.x * dim.y * dim.z);
  liong::log::info("created framebuffer (width=", dim.x, ", height=", dim.y,
    ", depth=", dim.z, ")");
  return Framebuffer { fmt, dim, framebuf_devmem };
}
void destroy_framebuf(Framebuffer& framebuf) {
  if (framebuf.framebuf_devmem.alloc_base) {
    free_mem(framebuf.framebuf_devmem);
  }
  framebuf = {};
  liong::log::info("destroyed framebuffer");
}

Mesh create_mesh(const MeshConfig& mesh_cfg) {
  auto vert_buf_offset = 0;
  auto vert_buf_size = mesh_cfg.nvert * mesh_cfg.vert_stride;

  auto idx_buf_offset = vert_buf_offset +
    align_addr(vert_buf_size, L_OPTIMAL_DEVMEM_ALIGN);
  auto idx_buf_size = mesh_cfg.ntri * mesh_cfg.tri_stride;

  auto pretrans_offset = idx_buf_offset +
    align_addr(idx_buf_size, OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT);
  auto pretrans_size = sizeof(mesh_cfg.pretrans);

  ASSERT << (vert_buf_size != 0)
    << "vertex buffer size cannot be zero";
  ASSERT << (idx_buf_size != 0)
    << "triangle index buffer size cannot be zero";

  auto alloc_size = pretrans_offset + pretrans_size;
  auto devmem = alloc_mem(alloc_size,
    std::max(L_OPTIMAL_DEVMEM_ALIGN, OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT));

  auto vert_slice = devmem.slice(vert_buf_offset, vert_buf_size);
  auto idx_slice = devmem.slice(idx_buf_offset, idx_buf_size);
  auto pretrans_slice = devmem.slice(pretrans_offset, pretrans_size);
  upload_mem(mesh_cfg.vert_buf, vert_slice, vert_buf_size);
  upload_mem(mesh_cfg.idx_buf, idx_slice, idx_buf_size);
  upload_mem(&mesh_cfg.pretrans, pretrans_slice, pretrans_size);

  OptixBuildInput build_in {};
  build_in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  build_in.triangleArray.vertexBuffers = new CUdeviceptr { vert_slice.ptr };
  build_in.triangleArray.vertexFormat = mesh_cfg.vert_fmt;
  build_in.triangleArray.numVertices = mesh_cfg.nvert;
  build_in.triangleArray.vertexStrideInBytes = mesh_cfg.vert_stride;
  build_in.triangleArray.indexBuffer = idx_slice.ptr;
  build_in.triangleArray.indexFormat = mesh_cfg.idx_fmt;
  build_in.triangleArray.numIndexTriplets = mesh_cfg.ntri;
  build_in.triangleArray.indexStrideInBytes = mesh_cfg.tri_stride;
  build_in.triangleArray.preTransform = pretrans_slice.ptr;
#if OPTIX_VERSION > 70000
  // OptiX 7.1 and higher
  build_in.triangleArray.transformFormat =
    OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
#endif // OPTIX_VERSION > 70000
  //build_in.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
  build_in.triangleArray.flags = new uint32_t[1] {};
  build_in.triangleArray.numSbtRecords = 1;

  return Mesh {
    devmem,
    vert_slice,
    idx_slice,
    pretrans_slice,
    build_in,
  };
}
void destroy_mesh(Mesh& mesh) {
  if (mesh.build_in.triangleArray.flags) {
    delete[] mesh.build_in.triangleArray.flags;
  }
  if (mesh.build_in.triangleArray.vertexBuffers) {
    delete[] mesh.build_in.triangleArray.vertexBuffers;
  }
  free_mem(mesh.devmem);
  mesh = {};
  liong::log::info("destroyed mesh");
}



SceneObject create_sobj() {
  return SceneObject { new AsFeedback {} };
}
void destroy_sobj(SceneObject& sobj) {
  if (sobj.inner) {
    free_mem(sobj.inner->devmem);
    delete sobj.inner;
  }
  sobj = {};
  liong::log::info("destroyed scene object");
}



Scene create_scene() {
  return Scene { new AsFeedback {}, {} };
}
void destroy_scene(Scene& scene) {
  free_mem(scene.inner->devmem);
  if (scene.inner) { delete scene.inner; }
  scene = {};
  liong::log::info("destroyed scene");
}
void add_scene_sobj(
  Scene& scene,
  const SceneObject& sobj,
  const Transform& trans,
  const DeviceMemorySlice& mat_devmem
) {
  auto elem = SceneElement { sobj.inner, trans, mat_devmem };
  scene.elems.emplace_back(std::move(elem));
}



Transaction create_transact() {
  CUstream stream;
  CUDA_ASSERT << cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  return Transaction { stream, {}, {} };
}
void _free_managed_mem(Transaction& transact) {
  for (auto devmem : transact.mnged_devmems) {
    free_mem(devmem);
  }
  for (auto hostmem : transact.mnged_hostmems) {
    delete hostmem;
  }
  transact.mnged_devmems.clear();
  transact.mnged_hostmems.clear();
}
bool pool_transact(Transaction& transact) {
  auto res = cuStreamQuery(transact.stream);
  if (res == CUDA_SUCCESS) {
    _free_managed_mem(transact);
    return true;
  } else if (res == CUDA_ERROR_NOT_READY) {
    return false;
  }
  CUDA_ASSERT << res;
}
void wait_transact(Transaction& transact) {
  CUDA_ASSERT << cuStreamSynchronize(transact.stream);
  _free_managed_mem(transact);
}
void destroy_transact(Transaction& transact) {
  if (transact.stream) { CUDA_ASSERT << cuStreamDestroy(transact.stream); }
  _free_managed_mem(transact);
  transact = {};
  liong::log::info("finalized transaction");
}



void manage_devmem(Transaction& transact, DeviceMemory&& devmem) {
  transact.mnged_devmems.emplace_back(std::forward<DeviceMemory>(devmem));
}
void manage_hostmem(Transaction& transact, void* hostmem) {
  transact.mnged_hostmems.emplace_back(hostmem);
}
void cmd_transfer_mem(
  Transaction& transact,
  const DeviceMemorySlice& src,
  const DeviceMemorySlice& dst
) {
  ASSERT << (src.size <= dst.size)
    << "transfer out of range";
  ASSERT << (((src.ptr < dst.ptr) && (src.ptr + src.size <= dst.ptr)) ||
    ((src.ptr > dst.ptr) && (src.ptr >= dst.ptr + dst.size)))
    << "transfer range overlapped";
  CUDA_ASSERT << cuMemcpyAsync(dst.ptr, src.ptr, src.size, transact.stream);
}
void cmd_upload_mem(
  Transaction& transact,
  const void* src,
  const DeviceMemorySlice& dst,
  size_t size
) {
  ASSERT << (size <= dst.size)
    << "memory read out of range";
  CUDA_ASSERT << cuMemcpyHtoDAsync(dst.ptr, src, size, transact.stream);
}
void cmd_download_mem(
  Transaction& transact,
  const DeviceMemorySlice& src,
  void* dst,
  size_t size
) {
  CUDA_ASSERT << cuMemcpyDtoHAsync(dst, src.ptr, size, transact.stream);
}

void cmd_traverse(
  Transaction& transact,
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  uint3 launch_size
) {
  OPTIX_ASSERT << optixLaunch(pipe.pipe, transact.stream,
    pipe_data.launch_cfg_devmem.ptr, pipe_data.launch_cfg_devmem.size,
    &pipe_data.sbt, launch_size.x, launch_size.y, launch_size.z);
  liong::log::info("scheduled transaction for scene traversal");
}


void cmd_init_pipe_data(
  Transaction& transact,
  const Pipeline& pipe,
  const PipelineData& pipe_data
) {
  const auto& pipe_layout = pipe.pipe_layout;
  auto i = 0;
  OptixShaderBindingTable sbt {};
  const auto base = pipe_data.sbt_devmem.ptr;
  uint8_t* sbt_hostbuf = new uint8_t[pipe.pipe_layout.sbt_size] {};

#define L_FILL_SBT_SINGLE_DATA(optix_name, prep_name, cfg_name)                \
  if (pipe_layout.sbt_##prep_name##_stride) {                                  \
    auto hbase = sbt_hostbuf + pipe_layout.sbt_##prep_name##_offset;           \
    OPTIX_ASSERT << optixSbtRecordPackHeader(pipe.pgrps[i], hbase);            \
    ++i;                                                                       \
  }
#define L_FILL_SBT_MULTI_DATA(optix_name, prep_name, cfg_name)                 \
  for (auto j = 0; j < pipe_layout.nsbt_##prep_name; ++j) {                    \
    auto hbase = sbt_hostbuf + pipe_layout.sbt_##prep_name##_offset +          \
      pipe_layout.sbt_##prep_name##_stride * j;                                \
    OPTIX_ASSERT << optixSbtRecordPackHeader(pipe.pgrps[i], hbase);            \
    ++i;                                                                       \
  }
#define L_FILL_SBT_INSTANCED_MULTI_DATA(optix_name, prep_name, cfg_name)       \
  for (auto j = 0; j < pipe_layout.nsbt_##prep_name; ++j) {                    \
    for (auto k = 0; k < pipe_layout.max_ninst; ++k) {                         \
      auto hbase = sbt_hostbuf + pipe_layout.sbt_##prep_name##_offset +        \
        pipe_layout.sbt_##prep_name##_stride *                                 \
        (k * pipe_layout.nsbt_##prep_name + j);                                \
      OPTIX_ASSERT << optixSbtRecordPackHeader(pipe.pgrps[i], hbase);          \
    }                                                                          \
    ++i;                                                                       \
  }

  // ORDER IS IMPORTANT; DO NOT RESORT >>>
  L_FILL_SBT_SINGLE_DATA(raygen, raygen, rg);
  L_FILL_SBT_SINGLE_DATA(exception, except, ex);
  L_FILL_SBT_MULTI_DATA(miss, miss, ms);
  L_FILL_SBT_INSTANCED_MULTI_DATA(hitgroup, hitgrp, hitgrp);
  L_FILL_SBT_MULTI_DATA(callables, call, dc);
  L_FILL_SBT_MULTI_DATA(callables, call, cc);
  // <<< ORDER IS IMPORTANT; DO NOT RESORT

  cmd_upload_mem(transact, sbt_hostbuf, pipe_data.sbt_devmem,
    pipe_layout.sbt_size);
  manage_hostmem(transact, sbt_hostbuf);

  liong::log::info("scheduled pipeline data initialization");
}


void _cmd_build_as(
  Transaction& transact,
  const Context& ctxt,
  const OptixBuildInput& build_in,
  AsFeedback* as_fb,
  bool can_compact
) {
  const size_t BUILD_PROP_OFFSET = 0;
  const size_t BUILD_PROP_DL_SIZE = sizeof(OptixAabb);
  const size_t BUILD_PROP_COMPACT_DL_SIZE = BUILD_PROP_DL_SIZE + sizeof(size_t);
  const size_t TRAV_OFFSET = BUILD_PROP_OFFSET + BUILD_PROP_COMPACT_DL_SIZE;

  OptixAccelBuildOptions build_opt = {
    // Otherwise `optixGetTriangleVertexData` won't work. Two hours have been
    // wasted here.
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
    // TODO: (penguinliong) Update, compaction, build/trace speed preference.
    (can_compact ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : OPTIX_BUILD_FLAG_NONE),
    OPTIX_BUILD_OPERATION_BUILD,
    // Motion not supported.
    OptixMotionOptions { 0, OPTIX_MOTION_FLAG_NONE, 0.0, 0.0 }
  };

  OptixAccelBufferSizes buf_size;
  OPTIX_ASSERT << optixAccelComputeMemoryUsage(ctxt.optix_dc, &build_opt,
    &build_in, 1, &buf_size);

  auto build_prop_size = can_compact ?
    sizeof(OptixAabb) + sizeof(size_t) : sizeof(OptixAabb);

  auto out_devmem = alloc_mem(buf_size.outputSizeInBytes,
      OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  auto temp_devmem = alloc_mem(buf_size.tempSizeInBytes + build_prop_size,
      OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  auto build_prop_devmem = temp_devmem.slice(buf_size.tempSizeInBytes);

  std::array<OptixAccelEmitDesc, 2> as_desc = {
    OptixAccelEmitDesc {
      build_prop_devmem.ptr,
      OPTIX_PROPERTY_TYPE_AABBS,
    },
    OptixAccelEmitDesc {
      build_prop_devmem.ptr + sizeof(OptixAabb),
      OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
    }
  };
  OPTIX_ASSERT << optixAccelBuild(ctxt.optix_dc, transact.stream, &build_opt,
    &build_in, 1, temp_devmem.ptr, buf_size.tempSizeInBytes, out_devmem.ptr,
    buf_size.outputSizeInBytes, &as_fb->trav, as_desc.data(),
    can_compact ? 2 : 1);
  as_fb->devmem = out_devmem;
  // Release temporary allocation automatically.
  manage_devmem(transact, std::move(temp_devmem));
  // Get AS build properties.
  cmd_download_mem(transact, build_prop_devmem, as_fb, build_prop_size);
}
void cmd_build_sobj(
  Transaction& transact,
  const Context& ctxt,
  const Mesh& mesh,
  SceneObject& sobj,
  bool can_compact
) {
  // TODO (penguinliong): Check for already allocated output memory.
  _cmd_build_as(transact, ctxt, mesh.build_in, sobj.inner, can_compact);
  liong::log::info("scheduled scene object build");
}


void cmd_build_scene(
  Transaction& transact,
  const Context& ctxt,
  Scene& scene,
  bool can_compact
) {
  auto ninst = scene.elems.size();
  OptixInstance* insts = new OptixInstance[ninst] {};
  for (auto i = 0; i < scene.elems.size(); ++i) {
    auto& inst = insts[i];
    std::memcpy(inst.transform, &scene.elems[i].trans, sizeof(inst.transform));
    inst.instanceId = i;
    // TODO: (penguinliong) We only allow one type of ray to be casted at a time
    // currently... and I think even if we want to cast multiple types of rays
    // we can stuff all those useful data into a single SBT record?
    inst.sbtOffset = i;
    // TODO: (penguinliong) Do we need instance layering for rays?
    // WARNING: DO NOT use `~0` instead of 255. 2 of my precious hours has been
    // wasted here. That's brutal.
    inst.visibilityMask = 255;
    inst.flags = OPTIX_INSTANCE_FLAG_NONE;
    //inst.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
    inst.traversableHandle = scene.elems[i].asfb->trav;
  }
  auto insts_devmem = shadow_mem(insts, sizeof(OptixInstance) * ninst,
    OPTIX_INSTANCE_BYTE_ALIGNMENT);
  delete[] insts;

  OptixBuildInput build_in {};
  build_in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  build_in.instanceArray.instances = insts_devmem.ptr;
  build_in.instanceArray.numInstances = ninst;

  _cmd_build_as(transact, ctxt, build_in, scene.inner, can_compact);
  // TODO (penguinliong): Check for already allocated output memory.
  manage_devmem(transact, std::move(insts_devmem));
  liong::log::info("scheduled scene build");
}


void cmd_compact_mem(
  Transaction& transact,
  const Context& ctxt,
  AsFeedback& as_fb
) {
  ASSERT << (as_fb.compact_size != 0)
    << "not a compactable as";
  if (as_fb.compact_size >= as_fb.devmem.size) {
    liong::log::info("ignored compaction since memory use will not decrease");
  }
  OptixTraversableHandle uncompact_trav = std::exchange(as_fb.trav, {});
  DeviceMemory uncompact_devmem = std::exchange(
    as_fb.devmem,
    alloc_mem(as_fb.compact_size, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)
  );

  OPTIX_ASSERT << optixAccelCompact(ctxt.optix_dc, transact.stream,
    uncompact_trav, as_fb.devmem.ptr, as_fb.devmem.size, &as_fb.trav);
  manage_devmem(transact, std::move(uncompact_devmem));
  liong::log::info("scheduled memory compaction (", uncompact_devmem.size,
    "bytes -> ", as_fb.compact_size, "bytes)");
}





//
// Extensions.
//

namespace ext {

std::vector<char> read_ptx(const char* ptx_path) {
  std::fstream fs(ptx_path, std::ios::in | std::ios::ate);
  ASSERT << fs.is_open()
    << "cannot open ptx file";
  size_t n = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::vector<char> buf;
  buf.resize(n);
  fs.read(buf.data(), n);
  fs.close();
  return buf;
}

Pipeline create_naive_pipe(
  const Context& ctxt,
  const NaivePipelineConfig& naive_pipe_cfg
) {
  PipelineConfig pipe_cfg{};
  pipe_cfg.debug = naive_pipe_cfg.debug;
  pipe_cfg.ptx_data = naive_pipe_cfg.ptx_data;
  pipe_cfg.ptx_size = naive_pipe_cfg.ptx_size;
  pipe_cfg.launch_cfg_name = "cfg";
  pipe_cfg.npayload_wd = 2;
  pipe_cfg.nattr_wd = 2;
  pipe_cfg.trace_depth = naive_pipe_cfg.trace_depth;
  pipe_cfg.max_ninst = naive_pipe_cfg.max_ninst;
  pipe_cfg.rg_cfg = {
    PipelineStageConfig {
      naive_pipe_cfg.rg_name,
      naive_pipe_cfg.ray_prop_size,
    }
  };
  pipe_cfg.ms_cfgs = {
    PipelineStageConfig {
      naive_pipe_cfg.ms_name,
      naive_pipe_cfg.env_size
    }
  };
  pipe_cfg.hitgrp_cfgs = {
    PipelineHitGroupConfig {
      nullptr,
      naive_pipe_cfg.ah_name,
      naive_pipe_cfg.ch_name,
      naive_pipe_cfg.mat_size
    }
  };
  pipe_cfg.launch_cfg_size = naive_pipe_cfg.launch_cfg_size;
  return create_pipe(ctxt, pipe_cfg);
}


std::vector<Mesh> import_meshes_from_file(const char* path) {
#ifdef L_USE_ASSIMP
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
#else
  throw std::logic_error("assimp must be linked to use this function");
#endif // L_USE_ASSIMP
}

void snapshot_hostmem(
  const void* hostmem,
  size_t hostmem_size,
  const void* head,
  size_t head_size,
  const char* path
) {
  if (hostmem_size == 0) {
    liong::log::warn("taking a snapshot of zero size");
  }
  std::fstream f(path, std::ios::out | std::ios::binary);
  f.write((const char*)head, head_size);
  f.write((const char*)hostmem, hostmem_size);
  f.close();
  liong::log::info("took a buffer snapshot to file: ", path);
}
void snapshot_devmem(
  const DeviceMemorySlice& devmem,
  const void* head,
  size_t head_size,
  const char* path
) {
  auto temp = std::malloc(devmem.size);
  download_mem(devmem, temp, devmem.size);
  snapshot_hostmem(temp, devmem.size, head, head_size, path);
  std::free(temp);
}
void snapshot_devmem(const DeviceMemorySlice& devmem, const char* path) {
  snapshot_devmem(devmem, nullptr, 0, path);
}

CommonSnapshot import_common_snapshot(const char* path) {
  std::fstream f(path, std::ios::in | std::ios::binary);
  SnapshotCommonHeader head;
  ASSERT << (f.readsome((char*)&head, sizeof(head)) == sizeof(head))
    << "unexpected eof";
  
  const char* magic_str = (const char*)&head.magic;
  bool is_le;
  if (magic_str[0] == '3' && magic_str[1] == 'J' && magic_str[2] == 'L' && magic_str[3] == 0x89) {
    is_le = true;
  } else if (magic_str[0] == 0x89 && magic_str[1] == 'L' && magic_str[2] == 'J' && magic_str[3] == '3') {
    is_le = false;
  } else {
    ASSERT << false
      << "imported file is not a common snapshot";
  }

  auto data = std::malloc(head.size);
  ASSERT << (f.readsome((char*)&data, head.size) == head.size)
    << "snapshot content is shorter than declared";
  
  return CommonSnapshot { data, head.size, head.type, is_le };
}
void destroy_common_snapshot(CommonSnapshot& snapshot) {
  std::free(snapshot.data);
  snapshot = {};
}

void _snapshot_framebuf_bmp(const Framebuffer& framebuf, std::fstream& f) {
  f.write("BM", 2);
  ASSERT << (framebuf.dim.z == 1)
    << "cannot take snapshot of 3d framebuffer";
  uint32_t img_size = framebuf.framebuf_devmem.size;
  uint32_t bmfile_hdr[] = { 14 + 108 + img_size, 0, 14 + 108 };
  f.write((const char*)bmfile_hdr, sizeof(bmfile_hdr));
  uint32_t bmcore_hdr[] = {
    108, // Size of header, here we use `BITMAPINFOHEADER`.
    framebuf.dim.x,
    framebuf.dim.y,
    1 | // Number of color planes.
    (32 << 16), // Bits per pixel.
    3, // Compression. (BI_BITFIELDS)
    img_size, // Raw image data size.
    2835, // Horizontal pixel per meter.
    2835, // Vertical pixel per meter.
    0, // (Unused) color palette count.
    0, // (Unused) important color count.
    0x000000FF, // Red channel mask.
    0x0000FF00, // Green channel mask.
    0x00FF0000, // Blue channel mask.
    0xFF000000, // Alpha channel mask.
    0x57696E20, // Color space. ("Win ")
    0,0,0,0,0,0,0,0,0, // CIEXYZTRIPLE end point.
    0, // Red gamma.
    0, // Green gamma.
    0, // Blue gamma.
  };
  f.write((const char*)bmcore_hdr, sizeof(bmcore_hdr));
  auto hostmem_size = framebuf.framebuf_devmem.size;
  auto hostmem = std::malloc(hostmem_size);
  download_mem(framebuf.framebuf_devmem, hostmem, hostmem_size);
  uint8_t buf;
  for (auto i = 0; i < framebuf.dim.y; ++i) {
    for (auto j = 0; j < framebuf.dim.x; ++j) {
      for (auto k = 0; k < framebuf.fmt.get_ncomp(); ++k) {
        buf = framebuf.fmt.extract(hostmem, i * framebuf.dim.x + j, k) * 255.99;
        f.write((const char*)&buf, sizeof(buf));
      }
    }
  }
  std::free(hostmem);
  f.flush();
  f.close();
}
void _snapshot_framebuf_exr(const Framebuffer& framebuf, std::fstream& f) {
  throw std::logic_error("not implemented yet");
}
FramebufferSnapshotFormat infer_snapshot_fmt(const char* path) {
  auto len = std::strlen(path);
  auto pos = path + len;
  auto c = '\0';
  while (pos-- != path) {
    c = *pos;
    if (c == '\\' || c == '/' || c == '.') { break; }
  }
  if (c != '.') {
    // There is no valid extension name for this file.
    return L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO;
  }
  pos++; // Omit the dot.
  if (std::strcmp(pos, "bmp") == 0) { return L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_BMP; }
  if (std::strcmp(pos, "exr") == 0) { return L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_EXR; }
  return L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO;
}
void snapshot_framebuf(
  const Framebuffer& framebuf,
  const char* path,
  FramebufferSnapshotFormat framebuf_snapshot_fmt
) {
  std::fstream f(path, std::ios::out | std::ios::binary);
  if (framebuf_snapshot_fmt == L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO) {
    framebuf_snapshot_fmt = infer_snapshot_fmt(path);
  }
  ASSERT << (framebuf_snapshot_fmt != L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_AUTO)
    << "cannot infer snapshot format";
  switch (framebuf_snapshot_fmt) {
  case L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_BMP:
    _snapshot_framebuf_bmp(framebuf, f);
    break;
  case L_EXT_FRAMEBUFFER_SNAPSHOT_FORMAT_EXR:
    _snapshot_framebuf_exr(framebuf, f);
    break;
  }
  f.close();
  liong::log::info("took snapshot of framebuffer to exr file: ", path);
}

} // namespace ext

} // namespace liong
