#pragma once
// Core functionalities of OptixLab.
// @PENGUINLIONG
#include "x.hpp"
#include "ty.hpp"

namespace liong {

// Initialize core functionalities of OptixLab.
extern void initialize();



extern Context create_ctxt(int dev_idx = 0);
extern void destroy_ctxt(Context& ctxt);



// Allocate at least `size` bytes of memory and ensure the base address is
// aligned to `align`.
extern DeviceMemory alloc_mem(size_t size, size_t align = 1);
// Free allocated memory.
extern void free_mem(DeviceMemory& devmem);
// Transfer data between two slices of CUDA device memory. `dst` MUST be able to
// contain an entire copy of content of `src`; `src` and `dst` MUST NOT overlap
// in range.
extern void transfer_mem(
  const DeviceMemorySlice& src,
  const DeviceMemorySlice& dst
);
// Upload data to CUDA device memory. `dst` MUST be able to contain an entire
// copy of content of `src`.
extern void upload_mem(
  const void* src,
  const DeviceMemorySlice& dst,
  size_t size
);
// Upload structured data to CUDA device memory.`dst` MUST be able to contain an
// entire copy of content of `src`.
template<typename TCont, typename TElem = typename TCont::value_type,
  typename _ = std::enable_if<type_traits::is_buffer_container_v<TCont>>>
void upload_mem(const TCont& src, const DeviceMemorySlice& dst) {
  upload_mem(src.data(), dst, sizeof(TElem) * src.size());
}
// Upload structured data to CUDA device memory.`dst` MUST be able to contain an
// entire copy of content of `src`.
template<typename T,
  typename _ = std::enable_if<type_traits::is_buffer_object_v<T>>>
void upload_mem(const T& src, const DeviceMemorySlice& dst) {
  upload_mem(&src, dst, sizeof(T));
}
// Download data from CUDA device memory. If the `size` of `dst` is shoter than
// the `src`, a truncated copy of `src` is downloaded.
extern void download_mem(const DeviceMemorySlice& src, void* dst, size_t size);
// Download structured data from CUDA device memory. If the size of `dst` is
// smaller than the `src`, a truncated copy of `src` is downloaded.
template<typename TCont, typename TElem = typename TCont::value_type,
  typename _ = std::enable_if<type_traits::is_buffer_container_v<TCont>>>
void download_mem(const DeviceMemorySlice& src, TCont& dst) {
  download_mem(src, dst.data(), sizeof(TElem) * dst.size());
}
// Download data from CUDA device memory. If the size of `dst` if smaller than
// the `src`, a truncated copy of `src` is downloaded.
template<typename T,
  typename _ = std::enable_if<type_traits::is_buffer_object_v<T>>>
void download_mem(const DeviceMemorySlice& src, T& dst) {
  download_mem(src, &dst, sizeof(T));
}
// Copy a slice of host memory to a new memory allocation on a device. The
// memory can be accessed globally by multiple streams.
extern DeviceMemory shadow_mem(const void* buf, size_t size, size_t align);
// Copy the content of `buf` to a new memory allocation on a device. The memory
// can be accessed globally by multiple streams.
template<typename TCont, typename TElem = typename TCont::value_type,
  typename _ = std::enable_if<type_traits::is_buffer_container_v<TCont>>>
DeviceMemory shadow_mem(const TCont& buf, size_t align = 1) {
  return shadow_mem(buf.data(), sizeof(TElem) * buf.size(), align);
}
template<typename T,
  typename _ = std::enable_if<type_traits::is_buffer_object_v<T>>>
DeviceMemory shadow_mem(const T& buf, size_t align = 1) {
  return shadow_mem(&buf, sizeof(T), align);
}



// Create ray-tracing pipeline from ptx file.
extern Pipeline create_pipe(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg
);
extern void destroy_pipe(Pipeline& pipe);



// Create pipeline data. PIpeline data MUST be initialized before use.
extern PipelineData create_pipe_data(const Pipeline& pipe);
extern void destroy_pipe_data(PipelineData& pipe);
extern DeviceMemorySlice slice_pipe_data(
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  OptixProgramGroupKind kind,
  uint32_t idx
);



extern Framebuffer create_framebuf(uint32_t w, uint32_t h, uint32_t d = 1);
extern void destroy_framebuf(Framebuffer& framebuf);
// Take a snapshot of the framebuffer content and write it to a BMP file.
//
// NOTE: Usual image app might not be able to read such 32-bit alpha-enabled
//       BMP but modern browsers seems supporting, at least on Firefox.
// WARNING: This only works properly on little-endian platforms.
extern void snapshot_framebuf(const Framebuffer& framebuf, const char* path);



// Create a mesh. The mesh data will be copied to the device side so it will be
// safe to release after mesh creation.
extern Mesh create_mesh(const MeshConfig& mesh_cfg, size_t mat_size = 0);
extern void destroy_mesh(Mesh& mesh);



// Create a scene object. The scene object is only traversable after being
// built.
extern SceneObject create_sobj();
extern void destroy_sobj(SceneObject& sobj);



// Create a scene from a set of `SceneObject`s. The scene is only traversable
// after being built. The parameter `SceneObject`s MUST have been built before
// becoming a child of a scene.
extern Scene create_scene(const std::vector<SceneObject>& sobjs);
extern void destroy_scene(Scene& scene);



// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
extern Transaction create_transact();
// Check if a transaction is finished. Returns `true` if the all previous
// commands are finished and `false` if the commands are still being processed.
// If the previous commands are all finished, the transaction is awaited.
extern bool pool_transact(Transaction& trans);
// Wait the transaction to complete. The transaction is awaited as the function
// returns.
extern void wait_transact(Transaction& trans);
// Release all related resources WITHOUT waiting it to finish.
extern void destroy_transact(Transaction& trans);

// Transfer ownership of a memory allocation to the given transaction. Any
// managed allocation will be freed automatically when the trasaction is found
// awaited.
extern void manage_mem(Transaction& transact, DeviceMemory&& devmem);

// The following functions are *command recording functions*. Command recording
// functions record asynchronous procedure in the given transaction. The time
// when the command gets executed is not guaranteed so the parameters MUST be
// kept alive until the transaction is awaited.

// Same as `transfer_mem`.
extern void cmd_transfer_mem(
  Transaction& transact,
  const DeviceMemorySlice& src,
  const DeviceMemorySlice& dst
);
// Same as `upload_mem`.
extern void cmd_upload_mem(
  Transaction& transact,
  const void* src,
  const DeviceMemorySlice& dst,
  size_t size
);
// Same as `download_mem`.
extern void cmd_download_mem(
  Transaction& transact,
  const DeviceMemorySlice& src,
  void* dst,
  size_t size
);

// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
extern void cmd_traverse(
  Transaction& transact,
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  const Framebuffer& framebuf,
  OptixTraversableHandle trav
);
// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
template<typename TTrav,
  typename _ = std::enable_if_t<std::is_same_v<
    decltype(std::remove_pointer_t<typename decltype(TTrav::inner)>::trav),
    OptixTraversableHandle>>>
void cmd_traverse(
  Transaction& transact,
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  const Framebuffer& framebuf,
  const TTrav& sobj
) {
  cmd_traverse(transact, pipe, pipe_data, framebuf, sobj.inner->trav);
}
// Initialize `PipelineData` layout. An uninitialized `PipelineData` cannot be
// correctly bound by its user pipeline and scheduling pipeline execution with
// invalid data will lead to undefined behavior.
void cmd_init_pipe_data(
  Transaction& transact,
  const Pipeline& pipe,
  const PipelineData& pipe_data
);
// Build a scene object. An unbuilt `SceneObject` cannot be correctly traversed
// and scheduling traversal on invalid traversable object will lead to undefined
// behavior.
extern void cmd_build_sobj(
  Transaction& transact,
  const Context& ctxt,
  const Mesh& mesh,
  SceneObject& sobj,
  bool can_compact = true
);
// Build a scene. An unbuilt `Scene` cannot be correctly traversed and
// scheduling traversal on invalid traversable object will lead to undefined
// behavior.
extern void cmd_build_scene(
  Transaction& transact,
  const Context& ctxt,
  Scene& scene,
  bool can_compact = true
);
// Compact traversable object memory. Memory compaction will invalidate all
// built references to the traversable object. The references MUST be re-built
// with corresponding `cmd_build_*` commands for referencing objects to return
// valid.
extern void cmd_compact_mem(
  Transaction& transact,
  const Context& ctxt,
  AsFeedback* as_fb
);
template<typename TTrav,
  typename _ = std::enable_if_t<std::is_same_v<
    decltype(std::remove_pointer_t<typename decltype(TTrav::inner)>::trav),
    OptixTraversableHandle>>>
void cmd_compact_mem(
  Transaction& transact,
  const Context& ctxt,
  const TTrav& x
) {
  cmd_compact_mem(transact, ctxt, x.inner);
}

}
