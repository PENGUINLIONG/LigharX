#define NOMINMAX
#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include "core.hpp"
#include "log.hpp"
#include "except.hpp"
// !!! MUST ONLY DEFINE ONCE !!!
#include <optix_function_table_definition.h>

namespace liong {


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
  OPTIX_ASSERT << optixDeviceContextDestroy(ctxt.optix_dc);
  ctxt = {};
  liong::log::info("destroyed optix context");
}


// Ensure a size is able to contain a `size` of content the even the content has
// to be aligned.
template<typename T,
  typename _ = std::enable_if<std::is_integral_v<T> || std::is_pointer_v<T>>>
  constexpr T align_size(T size, size_t align) {
  return size + align - 1;
}
// Align an `addr` (in either a pointer or a integer) to an given alignment.
template<typename T,
  typename _ = std::enable_if<std::is_integral_v<T> || std::is_pointer_v<T>>>
  constexpr T align_addr(T addr, size_t align) {
  return (T)(((size_t)addr - 1 + align) / align * align);
}


DeviceMemory alloc_mem(size_t size, size_t align) {
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
  CUDA_ASSERT << cuMemFree(devmem.alloc_base);
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
  ASSERT << (size <= dst.size)
    << "memory read out of range";
  CUDA_ASSERT << cuMemcpyHtoD(dst.ptr, src, size);
}
void download_mem(const DeviceMemorySlice& src, void* dst, size_t size) {
  CUDA_ASSERT << cuMemcpyDtoH(dst, src.ptr, size);
}

DeviceMemory shadow_mem(const void* buf, size_t size, size_t align) {
  auto devmem = alloc_mem(size, align);
  upload_mem(buf, devmem, size);
  return devmem;
}


// Read PTX representation from a single file specified in `pipe_cfg`.
std::vector<char> _read_ptx(const PipelineConfig& pipe_cfg) {
  std::fstream fs(pipe_cfg.mod_path, std::ios::in | std::ios::ate);
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
OptixModule _create_mod(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg,
  const std::vector<char>& ptx
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
    pipe_cfg.launch_param_name,
  };
  OptixModule mod;
  auto res = optixModuleCreateFromPTX(ctxt.optix_dc, &mod_opt, &pipe_opt,
    ptx.data(), ptx.size(), log, &log_len, &mod);
  if (log_len != 0 && res != OPTIX_SUCCESS) {
    liong::log::warn(log);
  }
  OPTIX_ASSERT << res;
  return mod;
}
struct PipelinePrep {
  std::vector<OptixProgramGroup> pgrps;

  size_t sbt_raygen_offset;

  size_t sbt_except_offset;

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
  std::vector<OptixProgramGroupDesc> pgrp_descs {};
  // Conservative size of device memory to contain all data referred by SBT.
  size_t sbt_size = 0, sbt_kind_max_size;
  // ORDER IS IMPORTANT; DO NOT RESORT >>>
  pipe_prep.sbt_raygen_offset = sbt_size;
  if (pipe_cfg.rg_cfg.name) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = pipe_cfg.rg_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    sbt_size += _sbt_align(pipe_cfg.rg_cfg.size);
  }

  pipe_prep.sbt_except_offset = sbt_size;
  if (pipe_cfg.ex_cfg.name) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = pipe_cfg.ex_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    sbt_size += _sbt_align(pipe_cfg.ex_cfg.size);
  }

  pipe_prep.sbt_miss_offset = sbt_size;
  pipe_prep.nsbt_miss = pipe_cfg.ms_cfgs.size();
  for (auto& ms_cfg : pipe_cfg.ms_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgrp_desc.raygen.module = mod;
    pgrp_desc.raygen.entryFunctionName = ms_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_prep.sbt_miss_stride =
      std::max(ms_cfg.size, _sbt_align(pipe_prep.sbt_miss_stride));
  }
  sbt_size += pipe_prep.sbt_miss_stride * pipe_prep.nsbt_miss;

  pipe_prep.sbt_hitgrp_offset = sbt_size;
  pipe_prep.nsbt_hitgrp = pipe_cfg.hitgrp_cfgs.size();
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
    pipe_prep.sbt_hitgrp_stride =
      std::max(hitgrp_cfg.size, _sbt_align(pipe_prep.sbt_hitgrp_stride));
  }
  sbt_size += pipe_prep.sbt_hitgrp_stride * pipe_prep.nsbt_hitgrp;

  pipe_prep.sbt_call_offset = sbt_size;
  pipe_prep.nsbt_call = pipe_cfg.dc_cfgs.size() + pipe_cfg.cc_cfgs.size();
  for (auto& dc_cfg : pipe_cfg.dc_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgrp_desc.callables.moduleDC = mod;
    pgrp_desc.callables.entryFunctionNameDC = dc_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_prep.sbt_call_stride =
      std::max(dc_cfg.size, _sbt_align(pipe_prep.sbt_call_stride));
  }
  for (auto& cc_cfg : pipe_cfg.cc_cfgs) {
    OptixProgramGroupDesc pgrp_desc {};
    pgrp_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgrp_desc.callables.moduleCC = mod;
    pgrp_desc.callables.entryFunctionNameCC = cc_cfg.name;
    pgrp_descs.push_back(pgrp_desc);
    pipe_prep.sbt_call_stride =
      std::max(cc_cfg.size, _sbt_align(pipe_prep.sbt_call_stride));
  }
  sbt_size += pipe_prep.sbt_call_stride * pipe_prep.nsbt_call;
  // <<< ORDER IS IMPORTANT; DO NOT RESORT

  OptixProgramGroupOptions opt;
  std::vector<OptixProgramGroup> pgrps;
  pgrps.resize(pgrp_descs.size());
  auto res = optixProgramGroupCreate(ctxt.optix_dc, pgrp_descs.data(),
    pgrp_descs.size(), &opt, log, &log_len,
    const_cast<OptixProgramGroup*>(pgrps.data()));
  if (log_len != 0 && res != OPTIX_SUCCESS) {
    liong::log::warn(log);
  }
  OPTIX_ASSERT << res;

  pipe_prep.pgrps = std::move(pgrps);
  pipe_prep.sbt_size = sbt_size;
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
    pipe_cfg.launch_param_name,
  };
  OptixPipelineLinkOptions link_opt {
    pipe_cfg.trace_depth,
    OPTIX_COMPILE_DEBUG_LEVEL_FULL,
    0,
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
void _set_pipe_stack_size(OptixPipeline pipe, const PipelineConfig& pipe_cfg) {
  const size_t DEFAULT_STACK_SIZE = 2048;
  const size_t DEFAULT_MAX_TRAV_GRAPH_DEPTH = 3;
  // TODO: (penguinliong) use stack size & max traversable depth depth hint from
  // pipeline config. These can be inferred from reveived PTX.
  OPTIX_ASSERT << optixPipelineSetStackSize(
    pipe,
    DEFAULT_STACK_SIZE,
    DEFAULT_STACK_SIZE,
    DEFAULT_STACK_SIZE,
    DEFAULT_MAX_TRAV_GRAPH_DEPTH
  );
}
struct SbtPrep {
  OptixShaderBindingTable sbt;
  DeviceMemory sbt_devmem;
};
SbtPrep _create_sbt_prep(
  const PipelineConfig& pipe_cfg,
  const PipelinePrep& pipe_prep
) {
  auto i = 0;
  OptixShaderBindingTable sbt {};
  DeviceMemory sbt_devmem =
    alloc_mem(pipe_prep.sbt_size, OPTIX_SBT_RECORD_ALIGNMENT);
  const auto base = sbt_devmem.ptr;
  uint8_t* sbt_hostbuf = new uint8_t[pipe_prep.sbt_size];

#define L_FILL_SBT_SINGLE_ENTRY_AND_DATA(optix_name, prep_name, cfg_name)      \
  if (pipe_cfg.cfg_name##_cfg.name) {                                          \
    sbt.optix_name##Record = base + pipe_prep.sbt_##prep_name##_offset;        \
    auto hbase = sbt_hostbuf + pipe_prep.sbt_##prep_name##_offset;             \
    OPTIX_ASSERT << optixSbtRecordPackHeader(pipe_prep.pgrps[i], hbase);       \
    std::memcpy(                                                               \
      hbase + OPTIX_SBT_RECORD_HEADER_SIZE,                                    \
      pipe_cfg.cfg_name##_cfg.data,                                            \
      pipe_cfg.cfg_name##_cfg.size                                             \
    );                                                                         \
    ++i;                                                                       \
  }

#define L_FILL_SBT_MULTI_ENTRY(optix_name, prep_name)                          \
  if (pipe_prep.nsbt_##prep_name) {                                            \
    sbt.optix_name##RecordBase = base + pipe_prep.sbt_##prep_name##_offset;    \
    sbt.optix_name##RecordStrideInBytes = pipe_prep.sbt_##prep_name##_stride;  \
    sbt.optix_name##RecordCount = pipe_prep.nsbt_##prep_name;                  \
  }

#define L_FILL_SBT_MULTI_DATA(optix_name, prep_name, cfg_name)                 \
  for (auto j = 0; j < pipe_prep.nsbt_##prep_name; ++j) {                      \
    auto hbase = sbt_hostbuf + pipe_prep.sbt_##prep_name##_offset +            \
      pipe_prep.sbt_##prep_name##_stride * j;                                  \
    OPTIX_ASSERT << optixSbtRecordPackHeader(pipe_prep.pgrps[i], hbase);       \
    auto& cfg = pipe_cfg.cfg_name##_cfgs.at(j);                                \
    std::memcpy(hbase + OPTIX_SBT_RECORD_HEADER_SIZE, cfg.data, cfg.size);     \
    ++i;                                                                       \
  }

  // ORDER IS IMPORTANT; DO NOT RESORT >>>
  L_FILL_SBT_SINGLE_ENTRY_AND_DATA(raygen, raygen, rg);
  L_FILL_SBT_SINGLE_ENTRY_AND_DATA(exception, except, ex);
  L_FILL_SBT_MULTI_ENTRY(miss, miss);
  L_FILL_SBT_MULTI_DATA(miss, miss, ms);
  L_FILL_SBT_MULTI_ENTRY(hitgroup, hitgrp);
  L_FILL_SBT_MULTI_DATA(hitgroup, hitgrp, hitgrp);
  L_FILL_SBT_MULTI_ENTRY(callables, call);
  L_FILL_SBT_MULTI_DATA(callables, call, dc);
  L_FILL_SBT_MULTI_DATA(callables, call, cc);
  // <<< ORDER IS IMPORTANT; DO NOT RESORT

#undef L_FILL_SBT_SINGLE_ENTRY_AND_DATA
#undef L_FILL_SBT_MULTI_ENTRY
#undef L_FILL_SBT_MULTI_DATA

  upload_mem(sbt_hostbuf, sbt_devmem, pipe_prep.sbt_size);
  delete[] sbt_hostbuf;
  liong::log::info("built sbt records");
  return SbtPrep { sbt, sbt_devmem };
}

Pipeline create_pipe(const Context& ctxt, const PipelineConfig& pipe_cfg) {
  auto ptx = _read_ptx(pipe_cfg);
  auto mod = _create_mod(ctxt, pipe_cfg, ptx);
  auto pipe_prep = _create_pipe_prep(ctxt, pipe_cfg, mod);
  auto pipe = _create_pipe(ctxt, pipe_cfg, pipe_prep.pgrps);
  auto sbt_prep = _create_sbt_prep(pipe_cfg, pipe_prep);
  std::stringstream ss;
  liong::log::info("created pipeline from module: ", pipe_cfg.mod_path);
  return Pipeline {
    mod,
    std::move(pipe_prep.pgrps),
    pipe,
    sbt_prep.sbt,
    std::move(sbt_prep.sbt_devmem)
  };
}
void destroy_pipe(Pipeline& pipe) {
  free_mem(pipe.sbt_devmem);
  OPTIX_ASSERT << optixPipelineDestroy(pipe.pipe);
  for (auto pgrp : pipe.pgrps) {
    OPTIX_ASSERT << optixProgramGroupDestroy(pgrp);
  }
  OPTIX_ASSERT << optixModuleDestroy(pipe.mod);
  pipe = Pipeline {};
  liong::log::info("destroyed pipeline");
}


Framebuffer create_framebuf(uint32_t width, uint32_t height, uint32_t depth) {
  ASSERT << ((width != 0) && (height != 0) && (depth != 0))
    << "framebuffer size cannot be zero";
  auto framebuf_devmem = alloc_mem(sizeof(uint32_t) * width * height * depth);
  liong::log::info("created framebuffer (width=", width, ", height=", height,
    ", depth=", depth, ")");
  return Framebuffer { width, height, depth, framebuf_devmem };
}
void destroy_framebuf(Framebuffer& framebuf) {
  free_mem(framebuf.framebuf_devmem);
  framebuf = {};
  liong::log::info("destroyed framebuffer");
}
void snapshot_framebuf(const Framebuffer& framebuf, const char* path) {
  std::fstream f(path, std::ios::out | std::ios::binary);
  f.write("BM", 2);
  ASSERT << (framebuf.depth == 1)
    << "cannot take snapshot of 3d framebuffer";
  uint32_t img_size = framebuf.framebuf_devmem.size;
  uint32_t bmfile_hdr[] = { 14 + 108 + img_size, 0, 14 + 108 };
  f.write((const char*)bmfile_hdr, sizeof(bmfile_hdr));
  uint32_t bmcore_hdr[] = {
    108, // Size of header, here we use `BITMAPINFOHEADER`.
    framebuf.width,
    framebuf.height,
    1 | // Number of color planes.
    (32 << 16), // Bits per pixel.
    3, // Compression. (BI_BITFIELDS)
    img_size, // Raw image data size.
    2835, // Horizontal pixel per meter.
    2835, // Vertical pixel per meter.
    0, // (Unused) color palette count.
    0, // (Unused) important color count.
    0x00FF0000, // Red channel mask.
    0x0000FF00, // Green channel mask.
    0x000000FF, // Blue channel mask.
    0xFF000000, // Alpha channel mask.
    0x57696E20, // Color space. ("Win ")
    0,0,0,0,0,0,0,0,0, // CIEXYZTRIPLE end point.
    0, // Red gamma.
    0, // Green gamma.
    0, // Blue gamma.
  };
  f.write((const char*)bmcore_hdr, sizeof(bmcore_hdr));
  auto hostmem_size = framebuf.framebuf_devmem.size;
  auto hostmem = new char[hostmem_size];
  download_mem(framebuf.framebuf_devmem, hostmem, hostmem_size);
  f.write(hostmem, hostmem_size);
  delete[] hostmem;
  f.close();
}

Mesh create_mesh(const MeshConfig& mesh_cfg) {
  auto vert_buf_size = mesh_cfg.nvert * mesh_cfg.vert_stride;
  auto idx_buf_size = mesh_cfg.ntri * mesh_cfg.tri_stride;

  ASSERT << (vert_buf_size != 0)
    << "vertex buffer size cannot be zero";
  ASSERT << (idx_buf_size != 0)
    << "triangle index buffer size cannot be zero";

  auto devmem = alloc_mem(vert_buf_size + idx_buf_size);
  auto vert_slice = devmem.slice(0, vert_buf_size);
  auto idx_slice = devmem.slice(vert_buf_size, idx_buf_size);
  upload_mem(mesh_cfg.vert_buf, vert_slice, vert_buf_size);
  upload_mem(mesh_cfg.idx_buf, idx_slice, idx_buf_size);

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
  // TODO: (penguinliong) Pre-transform has been ignored for now.
  build_in.triangleArray.flags = new uint32_t[1] {};
  build_in.triangleArray.numSbtRecords = 1;

  return Mesh { devmem, vert_slice, idx_slice, build_in };
}
void destroy_mesh(Mesh& mesh) {
  delete[] mesh.build_in.triangleArray.flags;
  delete mesh.build_in.triangleArray.vertexBuffers;
  free_mem(mesh.devmem);
  mesh = {};
}


SceneObject create_sobj() {
  return SceneObject { new AsFeedback {} };
}
void destroy_sobj(SceneObject& sobj) {
  free_mem(sobj.inner->devmem);
  delete sobj.inner;
  sobj = {};
}



Scene create_scene(const std::vector<SceneObject>& sobjs) {
  return Scene { new AsFeedback {}, sobjs };
}
void destroy_scene(Scene& scene) {
  free_mem(scene.inner->devmem);
  delete scene.inner;
  scene= {};
}
/*
struct DisassembledScene {
};
DisassembledScene disasm_scene(std::vector<>)
*/



Transaction create_transact() {
  CUstream stream;
  CUDA_ASSERT << cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  return Transaction { stream };
}
void _free_managed_mem(Transaction& transact) {
  for (auto devmem : transact.mnged_devmems) {
    free_mem(devmem);
  }
  transact.mnged_devmems.clear();
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
  CUDA_ASSERT << cuStreamDestroy(transact.stream);
  _free_managed_mem(transact);
  transact = {};
  liong::log::info("finalized transaction");
}



void manage_mem(Transaction& transact, DeviceMemory&& devmem) {
  transact.mnged_devmems.emplace_back(std::forward<DeviceMemory>(devmem));
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



}
