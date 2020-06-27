#include <optix_device.h>

#include "app.hpp"

namespace liong {

extern "C" __constant__ LaunchConfig cfg;

extern "C" __global__ void __closesthit__() {
}

extern "C" __global__ void __anyhit__() {
}

extern "C" __global__ void __miss__() {
}

extern "C" __global__ void __raygen__() {
  auto x = optixGetLaunchIndex().x;
  auto y = optixGetLaunchIndex().y;
  auto i = x + y * cfg.width;
  uint32_t color = 0xff000000 | i;

  cfg.framebuf[i] = color;
}

}
