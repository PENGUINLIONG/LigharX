#include <array>
#include "core.hpp"
#include "log.hpp"
#include "except.hpp"

namespace liong {


void cmd_traverse(
  Transaction& transact,
  const Pipeline& pipe,
  const Framebuffer& framebuf,
  OptixTraversableHandle trav
) {
  auto lparam = LaunchConfig {
    framebuf.width,
    framebuf.height,
    framebuf.depth,
    trav,
    reinterpret_cast<uint32_t*>(framebuf.framebuf_devmem.ptr)
  };
  DeviceMemory lparam_devmem = shadow_mem(lparam);

  optixLaunch(pipe.pipe, transact.stream, lparam_devmem.ptr, lparam_devmem.size,
    &pipe.sbt, framebuf.width, framebuf.height, 1);
  liong::log::info("initiated transaction for scene traversal");
  manage_mem(transact, std::move(lparam_devmem));
}


const size_t L_BLAS_BUILD_PROP_SIZE = sizeof(OptixAabb) + sizeof(size_t);
void cmd_build_sobj(
  Transaction& transact,
  const Context& ctxt,
  const Mesh& mesh,
  SceneObject& sobj_,
  bool allow_compact
) {
  auto& sobj = *sobj_.inner;
  OptixAccelBuildOptions build_opt = {
    // TODO: (penguinliong) Update, compaction, build/trace speed preference.
    OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
    OPTIX_BUILD_OPERATION_BUILD,
    // Motion not supported.
    OptixMotionOptions { 0, OPTIX_MOTION_FLAG_NONE, 0.0, 0.0 }
  };

  OptixAccelBufferSizes buf_size;
  OPTIX_ASSERT << optixAccelComputeMemoryUsage(ctxt.optix_dc, &build_opt,
    &mesh.build_in, 1, &buf_size);

  // TODO (penguinliong): Check for already allocated output memory.

  sobj.devmem = alloc_mem(buf_size.outputSizeInBytes,
    OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  auto temp_devmem = alloc_mem(buf_size.tempSizeInBytes +
    L_BLAS_BUILD_PROP_SIZE, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
  auto build_prop_devmem = temp_devmem.slice(buf_size.tempSizeInBytes);
  
  std::array<OptixAccelEmitDesc, 2> blas_desc = {
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
    &mesh.build_in, 1, temp_devmem.ptr, buf_size.tempSizeInBytes,
    sobj.devmem.ptr, buf_size.outputSizeInBytes, &sobj.trav, blas_desc.data(),
    2);
  // Release temporary allocation automatically.
  manage_mem(transact, std::move(temp_devmem));
  // Get AS build properties.
  cmd_download_mem(transact, build_prop_devmem, &sobj, L_BLAS_BUILD_PROP_SIZE);
}


void schedule_build_tlas() {

}


void cmd_compact_as(
  Transaction& transact,
  const Context& ctxt,
  SceneObject& sobj_
) {
  auto& sobj = *sobj_.inner;
  OptixTraversableHandle uncompact_trav = std::exchange(sobj.trav, {});
  DeviceMemory uncompact_devmem = std::exchange(
    sobj.devmem,
    alloc_mem(sobj.compact_size, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)
  );

  OPTIX_ASSERT << optixAccelCompact(ctxt.optix_dc, transact.stream,
    uncompact_trav, sobj.devmem.ptr, sobj.devmem.size, &sobj.trav);
  manage_mem(transact, std::move(uncompact_devmem));
}


}
