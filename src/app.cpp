#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include "app.hpp"
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
// !!! MUST ONLY DEFINE ONCE !!!
#include <optix_function_table_definition.h>

#undef ERROR
#undef max
#undef min

namespace {

void log_cb(liong::log::LogLevel lv, const std::string& msg) {
  using liong::log::LogLevel;
  switch (lv) {
  case LogLevel::INFO:
    printf("[\x1B[32mINFO\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::WARNING:
    printf("[\x1B[33mWARN\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::ERROR:
    printf("[\x1B[31mERROR\x1B[0m] %s\n", msg.c_str());
    break;
  }
}

}


class CudaException : public std::exception {
  std::string msg;
public:
  CudaException(CUresult code) {
    std::stringstream buf;
    const char* err_str;
    if (cuGetErrorString(code, &err_str) != CUDA_SUCCESS) {
      buf << "failed to describe error " << code;
    } else {
      buf << "cuda error: " << err_str;
    }
    msg = buf.str();
  }
  CudaException(cudaError_t code) {
    std::stringstream buf;
    const char* err_str = cudaGetErrorString(code);
    buf << "cuda runtime error: " << err_str;
    msg = buf.str();
  }

  const char* what() const override {
    return msg.c_str();
  }

};
struct CudaAssert {
  inline const CudaAssert& operator<<(CUresult code) const {
    if (code != CUDA_SUCCESS) { throw CudaException(code); }
    return *this;
  }
  inline const CudaAssert& operator<<(cudaError_t code) const {
    if (code != CUDA_SUCCESS) { throw CudaException(code); }
    return *this;
  }
} CUDA_ASSERT;

class OptixException : public std::exception {
  std::string msg;
public:
  OptixException(OptixResult code) {
    std::stringstream buf;
    const char* err_str = optixGetErrorString(code);
    buf << "optix error: " << err_str;
    msg = buf.str();
  }

  const char* what() const override {
    return msg.c_str();
  }
};
struct OptixAssert {
  inline const OptixAssert& operator<<(OptixResult code) const {
    if (code != CUDA_SUCCESS) { throw OptixException(code); }
    return *this;
  }
} OPTIX_ASSERT;

class AssertionFailedException : public std::exception {
  std::string msg;
public:
  AssertionFailedException(const std::string& msg) : msg(msg) {}

  const char* what() const override {
    return msg.c_str();
  }
};
struct Asserted { bool cond; };
struct Assert {} ASSERT;
inline const Assert operator<<(Asserted a, const std::string& msg) {
  if (a.cond) {
    return {};
  } else {
    throw AssertionFailedException { msg };
  }
}
inline const Asserted operator<<(Assert a, bool cond) {
  return Asserted { cond };
}


void initialize() {
  CUDA_ASSERT << cuInit(0);
  OPTIX_ASSERT << optixInit();
}

struct Context {
  CUdevice dev;
  CUcontext cuda_ctxt;
  OptixDeviceContext optix_dc;
};

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
Context create_ctxt(int dev_idx = 0) {
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

struct DeviceMemorySlice {
  CUdeviceptr ptr;
  size_t size;
};
struct DeviceMemory {
  // Base address of memory allocation.
  CUdeviceptr alloc_base;
  // Base address pointer aligned to specified alignment.
  CUdeviceptr ptr;
  // Size of (total) allocated memory including alignment padding.
  size_t alloc_size;
  // Size of allocated memory aligned to specified alignment.
  size_t size;
  operator DeviceMemorySlice() {
    return DeviceMemorySlice { ptr, size };
  }
  inline DeviceMemorySlice slice(size_t offset, size_t size) const {
    ASSERT << ((offset >= 0) && (offset + size <= this->size))
      << "slice out of range";
    return DeviceMemorySlice { ptr + offset, size };
  }
  inline DeviceMemorySlice slice(size_t offset) const {
    return slice(offset, size - offset);
  }
};

// Allocate at least `size` bytes of memory and ensure the base address is
// aligned to `align`.
DeviceMemory alloc_mem(size_t size, size_t align = 1) {
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
// Free allocated memory.
void free_mem(DeviceMemory& devmem) {
  CUDA_ASSERT << cuMemFree(devmem.alloc_base);
  liong::log::info("freed memory of ", devmem.size, " bytes");
  devmem = {};
}

// Transfer data between two slices of CUDA device memory. `dst` MUST be able to
// contain an entire copy of content of `src`; `src` and `dst` MUST NOT overlap
// in range.
void transfer_mem(DeviceMemorySlice src, DeviceMemorySlice dst) {
  ASSERT << (src.size <= dst.size)
    << "transfer out of range";
  ASSERT << (((src.ptr < dst.ptr) && (src.ptr + src.size <= dst.ptr)) ||
    ((src.ptr > dst.ptr) && (src.ptr >= dst.ptr + dst.size)))
    << "transfer range overlapped";
  CUDA_ASSERT << cuMemcpy(dst.ptr, src.ptr, src.size);
}
// Upload data to CUDA device memory. `dst` MUST be able to contain an entire
// copy of content of `src`.
void upload_mem(const void* src, DeviceMemorySlice dst, size_t size) {
  ASSERT << (size <= dst.size)
    << "memory read out of range";
  CUDA_ASSERT << cuMemcpyHtoD(dst.ptr, src, size);
}
// Upload structured data to CUDA device memory.`dst` MUST be able to contain an
// entire copy of content of `src`.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
  void upload_mem(const std::vector<T>& src, DeviceMemorySlice dst) {
  upload_mem(src.data(), dst, sizeof(T) * src.size());
}
// Upload structured data to CUDA device memory.`dst` MUST be able to contain an
// entire copy of content of `src`.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
  void upload_mem(const T& src, DeviceMemorySlice dst) {
  upload_mem(&src, dst, sizeof(T));
}
// Download data from CUDA device memory. If the `size` of `dst` is shoter than
// the `src`, a truncated copy of `src` is downloaded.
void download_mem(DeviceMemorySlice src, void* dst, size_t size) {
  CUDA_ASSERT << cuMemcpyDtoH(dst, src.ptr, size);
}
// Download structured data from CUDA device memory. If the size of `dst` is
// smaller than the `src`, a truncated copy of `src` is downloaded.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
void download_mem(DeviceMemorySlice src, std::vector<T>& dst) {
  download_mem(src, (void*)dst.data(), sizeof(T) * dst.size());
}
// Download data from CUDA device memory. If the size of `dst` if smaller than
// the `src`, a truncated copy of `src` is downloaded.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
void download_mem(DeviceMemorySlice src, T& dst) {
  download_mem(src, &dst, sizeof(T));
}

// Copy a slice of host memory to a new memory allocation on a device. The
// memory can be accessed globally by multiple streams.
DeviceMemory shadow_mem(const void* buf, size_t size, size_t align = 1) {
  auto devmem = alloc_mem(size, align);
  upload_mem(buf, devmem, size);
  return devmem;
}
// Copy the content of `buf` to a new memory allocation on a device. The memory
// can be accessed globally by multiple streams.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
DeviceMemory shadow_mem(const std::vector<T>& buf, size_t align = 1) {
  return shadow_mem(reinterpret_cast<const void*>(buf.data()),
    sizeof(T) * buf.size(), align);
}
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
DeviceMemory shadow_mem(const T& buf, size_t align = 1) {
  return shadow_mem(&buf, sizeof(T), align);
}



struct PipelineStageConfig {
  // Name of the stage function in provided module.
  const char* name;
  // Data used in a pipeline stage which will be copied to device memory.
  const void* data;
  // Size of `data` in bytes.
  size_t size;
};
struct PipelineHitGroupConfig {
  // Intersection stage function name.
  const char* is_name;
  // Any-hit stage function name.
  const char* ah_name;
  // Closest-hit stage function name.
  const char* ch_name;
  // Data used by a hit group which will be copied to device memory.
  const void* data;
  // Size of `data` in bytes.
  size_t size;
};

// All the data necessary in pipeline createion.
struct PipelineConfig {
  // Whether the pipeline is set up in debug mode. Extra debug information will
  // be provided if enabled.
  bool debug;

  // Path to the pipeline module containing all stages and callables of the
  // pipeline.
  const char* mod_path;
  // Launch parameter variable name. The parameter variable will be ignored if
  // this field is empty.
  const char* launch_param_name;
  // Number of words used for the payload. [0..8]
  int npayload_wd;
  // Number of words used for the attributes. [0..8]
  int nattr_wd;
  // Maximum trace recursion depth. [0..31]
  unsigned trace_depth;

  // Although it's POSSIBLE to use multiple raygen functions but for efficiency
  // here we DO NOT support it.
  PipelineStageConfig rg_cfg;
  PipelineStageConfig ex_cfg;
  std::vector<PipelineStageConfig> ms_cfgs;
  std::vector<PipelineHitGroupConfig> hitgrp_cfgs;
  std::vector<PipelineStageConfig> dc_cfgs;
  std::vector<PipelineStageConfig> cc_cfgs;
};

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
  optixPipelineSetStackSize(
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
    optixSbtRecordPackHeader(pipe_prep.pgrps[i], hbase);                       \
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
    optixSbtRecordPackHeader(pipe_prep.pgrps[i], hbase);                       \
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

// Pipeline-related opaque resources.
struct Pipeline {
  OptixModule mod;
  std::vector<OptixProgramGroup> pgrps;
  OptixPipeline pipe;
  OptixShaderBindingTable sbt;
  DeviceMemory sbt_devmem;
};
// Create ray-tracing pipeline from ptx file.
Pipeline create_pipe(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg
) {
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
  optixPipelineDestroy(pipe.pipe);
  for (auto pgrp : pipe.pgrps) {
    optixProgramGroupDestroy(pgrp);
  }
  optixModuleDestroy(pipe.mod);
  pipe = Pipeline {};
  liong::log::info("destroyed pipeline");
}

struct Transaction {
  CUstream stream;
  DeviceMemory lparam_devmem;
};

struct Framebuffer {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  DeviceMemory framebuf_devmem;
};

Framebuffer create_framebuf(
  uint32_t width,
  uint32_t height,
  uint32_t depth = 1
) {
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
// Take a snapshot of the framebuffer content and write it to a BMP file.
//
// NOTE: Usual image app might not be able to read such 32-bit alpha-enabled
//       BMP but modern browsers seems supporting, at least on Firefox.
// WARNING: This only works properly on little-endian platforms.
void snapshot_framebuf(Framebuffer& framebuf, const char* path) {
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

struct MeshConfig {
  // Vertex data buffer.
  const void* vert_buf;
  // Format of vertex data.
  OptixVertexFormat vert_fmt;
  // Number of vertices in the buffer.
  size_t nvert;
  // Stride between packs of vertex data.
  size_t vert_stride;

  // Index data buffer.
  const void* idx_buf;
  // Format of index tuple.
  OptixIndicesFormat idx_fmt;
  // Number of triangles in `idx_buf`.
  size_t ntri;
  // Stride between tuples of triangle vertex indices.
  size_t tri_stride;
};
struct Mesh {
  DeviceMemory devmem;
  DeviceMemorySlice vert_slice;
  DeviceMemorySlice idx_slice;
  OptixBuildInput build_in;
};
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

struct SceneObject {
  bool dirty;
  OptixTraversableHandle trav;
  OptixAabb aabb;
  // Vertex and index buffer. Vertex data is placed at first and index follows
  // that.
  DeviceMemory devmem;
};
SceneObject create_sobj(
  const Context& ctxt,
  const Mesh& mesh
) {
  const static OptixAccelBuildOptions build_opts = {
    // TODO: (penguinliong) Update, compaction, build/trace speed preference.
    OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
    OPTIX_BUILD_OPERATION_BUILD,
    // Motion not supported.
    OptixMotionOptions { 0, OPTIX_MOTION_FLAG_NONE, 0.0, 0.0 },
  };

  CUstream stream {};
  // TODO: (penguinliong) To command buffer.
  CUDA_ASSERT << cuStreamCreate(&stream, 0);

  OptixTraversableHandle blas;
  struct {
    OptixAabb aabb;
    size_t compact_size;
  } build_prop;

  OptixAccelBufferSizes buf_size;
  OPTIX_ASSERT << optixAccelComputeMemoryUsage(ctxt.optix_dc, &build_opts,
    &mesh.build_in, 1, &buf_size);
  auto out_devmem = alloc_mem(buf_size.outputSizeInBytes,
    OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  auto temp_devmem = alloc_mem(buf_size.tempSizeInBytes + sizeof(build_prop),
    OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  size_t build_feedback_devmem_base =
    temp_devmem.ptr + buf_size.tempSizeInBytes;
  std::array<OptixAccelEmitDesc, 2> blas_desc = {
    OptixAccelEmitDesc {
      build_feedback_devmem_base,
      OPTIX_PROPERTY_TYPE_AABBS,
    },
    OptixAccelEmitDesc {
      build_feedback_devmem_base + sizeof(OptixAabb),
      OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
    }
  };
  OPTIX_ASSERT << optixAccelBuild(ctxt.optix_dc, stream, &build_opts,
    &mesh.build_in, 1, temp_devmem.ptr, buf_size.tempSizeInBytes,
    out_devmem.ptr, buf_size.outputSizeInBytes, &blas, blas_desc.data(),
    blas_desc.size());

  // TODO: (penguinliong) To command buffer.
  CUDA_ASSERT << cuStreamSynchronize(stream);

  download_mem(temp_devmem.slice(buf_size.tempSizeInBytes), build_prop);
  free_mem(temp_devmem);

  auto compact_devmem = alloc_mem(build_prop.compact_size,
    OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  OPTIX_ASSERT << optixAccelCompact(ctxt.optix_dc, stream, blas,
    compact_devmem.ptr, compact_devmem.size, &blas);
  free_mem(out_devmem);

  // TODO: (penguinliong) To command buffer.
  CUDA_ASSERT << cuStreamDestroy(stream);

  return SceneObject { false, blas, build_prop.aabb, compact_devmem };
}
void destroy_sobj(SceneObject sobj) {
  free_mem(sobj.devmem);
  sobj = {};
}
/*
struct Scene {
  OptixTraversableHandle trav;
  std::vector<SceneObject> sobjs;
  DeviceMemory devmem;
};
struct DisassembledScene {

};
Scene create_scene(std::vector<SceneObject> sobjs) {
  
}
DisassembledScene disasm_scene(std::vector<>)
void destroy_scene(Scene& scene) {

}
*/

// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
template<typename TTrav,
  typename _ = std::enable_if_t<std::is_same_v<decltype(TTrav::trav),
    OptixTraversableHandle>>>
Transaction init_transact(
  Pipeline pipe,
  const Framebuffer& framebuf,
  const TTrav& sobj
) {
  auto lparam = LaunchConfig {
    framebuf.width,
    framebuf.height,
    framebuf.depth,
    sobj.trav,
    reinterpret_cast<uint32_t*>(framebuf.framebuf_devmem.ptr)
  };
  DeviceMemory lparam_devmem = shadow_mem(lparam);

  CUstream stream;
  CUDA_ASSERT << cuStreamCreate(&stream, CU_STREAM_DEFAULT);

  optixLaunch(pipe.pipe, stream, lparam_devmem.ptr, lparam_devmem.size,
    &pipe.sbt, framebuf.width, framebuf.height, 1);
  liong::log::info("initiated transaction for scene traversal");
  return Transaction { stream, std::move(lparam_devmem) };
}
// Check if a transaction is finished. Returns `true` if the transaction is
// finished and `false` if the transaction is still being computed.
bool pool_transact(const Transaction& trans) {
  auto res = cuStreamQuery(trans.stream);
  if (res == CUDA_SUCCESS) {
    return true;
  } else if (res == CUDA_ERROR_NOT_READY) {
    return false;
  }
  CUDA_ASSERT << res;
}
// Wait the transaction to complete.
void wait_transact(const Transaction& trans) {
  CUDA_ASSERT << cuStreamSynchronize(trans.stream);
}
// Release all related resources WITHOUT waiting it to finish.
void final_transact(Transaction& trans) {
  cuStreamDestroy(trans.stream);
  free_mem(trans.lparam_devmem);
  trans = {};
  liong::log::info("finalized transaction");
}





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
  Transaction transact;

  try {
    ctxt = create_ctxt();
    pipe = create_pipe(ctxt, pipe_cfg);
    framebuf = create_framebuf(32, 32);
    mesh = create_mesh(mesh_cfg);
    sobj = create_sobj(ctxt, mesh);
    transact = init_transact(pipe, framebuf, sobj);

    wait_transact(transact);

    snapshot_framebuf(framebuf, "./snapshot.bmp");

    liong::log::info("sounds good");
  } catch (const std::exception& e) {
    liong::log::error("application threw an exception");
    liong::log::error(e.what());
  } catch (...) {
    liong::log::error("application threw an illiterate exception");
  }
  final_transact(transact);
  destroy_sobj(sobj);
  destroy_mesh(mesh);
  destroy_framebuf(framebuf);
  destroy_pipe(pipe);
  destroy_ctxt(ctxt);

  liong::log::info("optix lab ended");

  return 0;
}

