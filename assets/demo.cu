#include <cmath>
#include "x.hpp"
#include "x-device.hpp"

namespace liong {

struct Material {
  float3 obj_color;
};
struct Environment {
  float3 sky_color;
};


SHADER_MAIN
void __closesthit__() {
  const auto& mat = *(const Material*)optixGetSbtDataPointer();
  auto pColor = (float3*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = mat.obj_color;
}

SHADER_MAIN
void __anyhit__() {
}

SHADER_MAIN
void __miss__() {
  const auto& env = *(const Environment*)optixGetSbtDataPointer();
  auto pColor = (float3*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  *pColor = env.sky_color;
}

SHADER_MAIN
void __raygen__() {
  float3 o = { 0, 0, -1 };
  Ray ray = perspect_ray(o);

  float3 color{};
  uint32_t wColor[] = PTR2WORDS(&color);

  optixTrace(cfg.trav, ray.o, ray.v,
    0.f, 1e20f, 0.0f, OptixVisibilityMask(255),
    // If you don't use it then YOU SHOULD DISABLE IT to bypass a program
    // invocation.
    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    0, 1, 0, wColor[0], wColor[1]);

  write_attm(color);
  //write_attm(color_encode_n1p1(ray.o));
}

}
