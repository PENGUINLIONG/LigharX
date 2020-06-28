#include "core.hpp"
#include "log.hpp"
#include "except.hpp"

namespace liong {

Transaction init_transact(
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

  CUstream stream;
  CUDA_ASSERT << cuStreamCreate(&stream, CU_STREAM_DEFAULT);

  optixLaunch(pipe.pipe, stream, lparam_devmem.ptr, lparam_devmem.size,
    &pipe.sbt, framebuf.width, framebuf.height, 1);
  liong::log::info("initiated transaction for scene traversal");
  return Transaction { stream, std::move(lparam_devmem) };
}
bool pool_transact(const Transaction& trans) {
  auto res = cuStreamQuery(trans.stream);
  if (res == CUDA_SUCCESS) {
    return true;
  } else if (res == CUDA_ERROR_NOT_READY) {
    return false;
  }
  CUDA_ASSERT << res;
}
void wait_transact(const Transaction& trans) {
  CUDA_ASSERT << cuStreamSynchronize(trans.stream);
}
void final_transact(Transaction& trans) {
  cuStreamDestroy(trans.stream);
  free_mem(trans.lparam_devmem);
  trans = {};
  liong::log::info("finalized transaction");
}

}
