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

  OPTIX_ASSERT << optixLaunch(pipe.pipe, transact.stream, lparam_devmem.ptr,
    lparam_devmem.size, &pipe.sbt, framebuf.width, framebuf.height, 1);
  liong::log::info("initiated transaction for scene traversal");
  manage_mem(transact, std::move(lparam_devmem));
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
    // TODO: (penguinliong) Update, compaction, build/trace speed preference.
    can_compact ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : OPTIX_BUILD_FLAG_NONE,
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
  manage_mem(transact, std::move(temp_devmem));
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
  auto ninst = scene.sobjs.size();
  OptixInstance* insts = new OptixInstance[ninst];
  for (auto i = 0; i < scene.sobjs.size(); ++i) {
    OptixInstance inst {};
    inst.transform[0] = 1.0f;
    inst.transform[5] = 1.0f;
    inst.transform[10] = 1.0f;
    inst.instanceId = i;
    // TODO: (penguinliong) Do we need instance layering for rays?
    // WARNING: DO NOT use `~0` instead of 255. 2 of my precious hours has been
    // wasted here. That's brutal.
    inst.visibilityMask = 255;
    inst.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
    inst.traversableHandle = scene.sobjs[i].inner->trav;
    insts[i] = std::move(inst);
  }
  auto insts_devmem = shadow_mem(insts, sizeof(OptixInstance) * ninst,
    OPTIX_INSTANCE_BYTE_ALIGNMENT);
  delete insts;

  OptixBuildInput build_in {};
  build_in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  build_in.instanceArray.instances = insts_devmem.ptr;
  build_in.instanceArray.numInstances = scene.sobjs.size();

  _cmd_build_as(transact, ctxt, build_in, scene.inner, can_compact);
  // TODO (penguinliong): Check for already allocated output memory.
  manage_mem(transact, std::move(insts_devmem));
  liong::log::info("scheduled scene build");
}


void cmd_compact_mem(
  Transaction& transact,
  const Context& ctxt,
  AsFeedback* as_fb
) {
  ASSERT << (as_fb->compact_size != 0)
    << "not a compactable as";
  OptixTraversableHandle uncompact_trav = std::exchange(as_fb->trav, {});
  DeviceMemory uncompact_devmem = std::exchange(
    as_fb->devmem,
    alloc_mem(as_fb->compact_size, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)
  );

  OPTIX_ASSERT << optixAccelCompact(ctxt.optix_dc, transact.stream,
    uncompact_trav, as_fb->devmem.ptr, as_fb->devmem.size, &as_fb->trav);
  manage_mem(transact, std::move(uncompact_devmem));
  liong::log::info("scheduled memory compaction");
}


}
