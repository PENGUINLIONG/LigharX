#include <optix_device.h>
#include "app.hpp"

namespace liong {

extern "C" __constant__ LaunchConfig cfg;

extern "C" __global__ void __closesthit__() {
  auto pColor = (uint32_t*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = 0xFFFF00FF;
}

extern "C" __global__ void __anyhit__() {
}

extern "C" __global__ void __miss__() {
  auto pColor = (uint32_t*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = 0;
}

extern "C" __global__ void __raygen__() {
  auto x = optixGetLaunchIndex().x;
  auto y = optixGetLaunchIndex().y;
  auto u = ((x + 0.5 - ((float)cfg.width / 2)) / cfg.width * 2.0);
  auto v = ((y + 0.5 - ((float)cfg.height / 2)) / cfg.height * 2.0);
  auto i = x + y * cfg.width;
  uint32_t color = 0;
  uint32_t wColor[] = PTR2WORDS(&color);

  optixTrace(cfg.trav, { u, v, 1.0 }, { 0.0, 0.0, -1.0 }, 0.f, 1e20f, 0.0f,
    OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
    wColor[0], wColor[1]);

  cfg.framebuf[i] = color;
}

}
