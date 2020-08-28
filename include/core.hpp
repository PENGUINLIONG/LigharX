#pragma once
// Core functionalities of LigharX.
// @PENGUINLIONG
#include "x.hpp"
#include "ty.hpp"

namespace liong {

// Initialize core functionalities of LigharX.
extern void initialize();



extern Context create_ctxt(int dev_idx = 0);
extern void destroy_ctxt(Context& ctxt);



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
// Download data from CUDA device memory. If the `size` of `dst` is shoter than
// the `src`, a truncated copy of `src` is downloaded.
extern void download_mem(const DeviceMemorySlice& src, void* dst, size_t size);
// Copy a slice of host memory to a new memory allocation on a device. The
// memory can be accessed globally by multiple streams.
extern DeviceMemory shadow_mem(const void* buf, size_t size, size_t align = 1);



extern Texture create_tex(const Context& ctxt, const TextureConfig& tex_cfg);
extern void destroy_tex(Texture& tex);



// Create ray-tracing pipeline from ptx file.
extern Pipeline create_pipe(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg
);
extern void destroy_pipe(Pipeline& pipe);



// Create pipeline data. Pipeline data MUST be initialized with
// `cmd_init_pipe_data` before use.
extern PipelineData create_pipe_data(const Pipeline& pipe);
extern void destroy_pipe_data(PipelineData& pipe);
extern DeviceMemorySlice slice_pipe_launch_cfg(
  const Pipeline& pipe,
  const PipelineData& pipe_data
);
extern DeviceMemorySlice slice_pipe_data(
  const Pipeline& pipe,
  const PipelineData& pipe_data,
  OptixProgramGroupKind kind,
  uint32_t idx
);


extern Framebuffer create_framebuf(
  PixelFormat fmt,
  uint3 dim
);
extern void destroy_framebuf(Framebuffer& framebuf);

extern Image create_img(
  PixelFormat fmt,
  uint3 dim
);
extern void destroy_img(Image& img);



// Create a mesh. The mesh data will be copied to the device side so it will be
// safe to release after mesh creation.
extern Mesh create_mesh(const MeshConfig& mesh_cfg);
extern void destroy_mesh(Mesh& mesh);



// Create a scene object. The scene object is only traversable after being
// built with `cmd_build_sobj`. Currently static scene object transformation is
// not supported. To traverse only one scene object with transformation you have
// to apply the transformation to the underlying meshes.
extern SceneObject create_sobj();
extern void destroy_sobj(SceneObject& sobj);



// Create a scene. The scene is only traversable after being built with
// `cmd_build_scene`.
extern Scene create_scene();
extern void destroy_scene(Scene& scene);
// Add a materialed scene object as a child of the scene. It is ALLOWED to add
// scene object to a scene while the object are still being built, but you MUST
// wait the building of scene objects to finish before you build the scene.
extern void add_scene_sobj(
  Scene& scene,
  const SceneObject& sobj,
  const Transform& trans,
  const DeviceMemorySlice& mat_buf
);



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
extern void manage_devmem(Transaction& transact, DeviceMemory&& devmem);
// Transfer ownership of a memory allocation to the given transaction. Any
// managed allocation will be freed automatically when the trasaction is found
// awaited. It should be noted that the destructor is guaranteed not running.
extern void manage_hostmem(Transaction& transact, void* hostmem);

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
  uint3 launch_size
);
// Initialize `PipelineData` layout. An uninitialized `PipelineData` CANNOT be
// correctly bound by its user pipeline and scheduling pipeline execution with
// invalid data will lead to undefined behavior. The user code MUST also upload
// SBT data AFTER the initialization.
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
  AsFeedback& as_fb
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
  cmd_compact_mem(transact, ctxt, *x.inner);
}
template<typename TTrav,
  typename _ = std::enable_if_t<std::is_same_v<
  decltype(std::remove_pointer_t<typename decltype(TTrav::inner)>::trav),
  OptixTraversableHandle>>>
void cmd_compact_mems(
  Transaction& transact,
  const Context& ctxt,
  const std::vector<TTrav>& xs
) {
  for (const auto& x : xs) {
    cmd_compact_mem(transact, ctxt, *x.inner);
  }
}

} // namespace liong
