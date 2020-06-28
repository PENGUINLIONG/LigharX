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
  typename _ = std::enable_if<!std::is_trivially_copyable_v<TCont> &
    std::is_same_v<decltype(TCont::size()), size_t> &
    std::is_trivially_copyable_v<TElem> &
    std::is_pointer_v<decltype(TCont::data)>>>
void upload_mem(const TCont& src, const DeviceMemorySlice& dst) {
  upload_mem(
    src.data(),
    dst,
    sizeof(TElem) * src.size()
  );
}
// Upload structured data to CUDA device memory.`dst` MUST be able to contain an
// entire copy of content of `src`.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
void upload_mem(const T& src, const DeviceMemorySlice& dst) {
  upload_mem(&src, dst, sizeof(T));
}
// Download data from CUDA device memory. If the `size` of `dst` is shoter than
// the `src`, a truncated copy of `src` is downloaded.
extern void download_mem(const DeviceMemorySlice& src, void* dst, size_t size);
// Download structured data from CUDA device memory. If the size of `dst` is
// smaller than the `src`, a truncated copy of `src` is downloaded.
template<typename TCont, typename TElem = typename TCont::value_type,
  typename _ = std::enable_if<!std::is_trivially_copyable_v<TCont> & 
    std::is_same_v<decltype(TCont::size()), size_t> &
    std::is_trivially_copyable_v<TElem> &
    std::is_pointer_v<decltype(TCont::data)>>>
void download_mem(const DeviceMemorySlice& src, TCont& dst) {
  download_mem(
    src,
    (void*)dst.data(),
    sizeof(TElem) * dst.size()
  );
}
// Download data from CUDA device memory. If the size of `dst` if smaller than
// the `src`, a truncated copy of `src` is downloaded.
template<typename T,
  typename _ = std::enable_if<std::is_trivially_copyable_v<T>>>
void download_mem(const DeviceMemorySlice& src, T& dst) {
  download_mem(src, &dst, sizeof(T));
}
// Copy a slice of host memory to a new memory allocation on a device. The
// memory can be accessed globally by multiple streams.
extern DeviceMemory shadow_mem(const void* buf, size_t size, size_t align = 1);
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



// Create ray-tracing pipeline from ptx file.
extern Pipeline create_pipe(
  const Context& ctxt,
  const PipelineConfig& pipe_cfg
);
extern void destroy_pipe(Pipeline& pipe);



extern Framebuffer create_framebuf(uint32_t w, uint32_t h, uint32_t d = 1);
extern void destroy_framebuf(Framebuffer& framebuf);
// Take a snapshot of the framebuffer content and write it to a BMP file.
//
// NOTE: Usual image app might not be able to read such 32-bit alpha-enabled
//       BMP but modern browsers seems supporting, at least on Firefox.
// WARNING: This only works properly on little-endian platforms.
extern void snapshot_framebuf(const Framebuffer& framebuf, const char* path);



extern Mesh create_mesh(const MeshConfig& mesh_cfg);
extern void destroy_mesh(Mesh& mesh);



extern SceneObject create_sobj(const Context& ctxt, const Mesh& mesh);
extern void destroy_sobj(SceneObject& sobj);



// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
extern Transaction init_transact(
  const Pipeline& pipe,
  const Framebuffer& framebuf,
  OptixTraversableHandle trav
);
// Create a CUDA stream and launch the stream for OptiX scene traversal
// controlled by the given pipeline.
//
// WARNING: `pipe` must be kept alive through out the lifetime of the created
// transaction.
template<typename TTrav,
  typename _ = std::enable_if_t<std::is_same_v<decltype(TTrav::trav),
    OptixTraversableHandle>>>
Transaction init_transact(
  const Pipeline& pipe,
  const Framebuffer& framebuf,
  const TTrav& sobj
) {
  return init_transact(pipe, framebuf, sobj.trav);
}
// Check if a transaction is finished. Returns `true` if the transaction is
// finished and `false` if the transaction is still being computed.
bool pool_transact(const Transaction& trans);
// Wait the transaction to complete.
void wait_transact(const Transaction& trans);
// Release all related resources WITHOUT waiting it to finish.
void final_transact(Transaction& trans);


}
