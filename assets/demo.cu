#include "x.hpp"
#include "x-mat.hpp"

namespace liong {

extern "C" __constant__ LaunchConfig cfg;

extern "C" __global__ void __closesthit__() {
  const auto& mat = *(const Material*)optixGetSbtDataPointer();
  auto pColor = (vec3*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = mat.obj_color;
}

extern "C" __global__ void __anyhit__() {
}

extern "C" __global__ void __miss__() {
  const auto& env = *(const Environment*)optixGetSbtDataPointer();
  auto pColor = (vec3*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = env.sky_color;
}

extern "C" __global__ void __raygen__() {
  auto x = optixGetLaunchIndex().x;
  auto y = optixGetLaunchIndex().y;
  auto u = ((float)(x * 2 + 1) / cfg.width - 1);
  auto v = ((float)(y * 2 + 1) / cfg.height - 1);
  auto i = x + y * cfg.width;
  vec3 color {};
  uint32_t wColor[] = PTR2WORDS(&color);

  optixTrace(cfg.trav, { u, v, 1.0 }, { 0.0, 0.0, -1.0 }, 0.f, 1e20f, 0.0f,
    OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
    wColor[0], wColor[1]);

  cfg.framebuf[i] = color.to_unorm_pack();
}

}
