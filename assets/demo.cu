#include <cmath>
#include "x.hpp"
#include "x-device.hpp"

namespace liong {

// Everything valuable during the lifetime of a ray.
template<typename TRes>
struct RayLife {
  // Incidental ray.
  Ray i;
  // Point of intersection (hit).
  float3 p;
  // Time-to-live. The number of remaining chances to launch traversal without
  // blasting the stack memory. It SHOULD be decresed before another traversal
  // and SHOULD be set to 0 on a miss.
  uint32_t ttl;
  // User-defined result data.
  TRes res;
};

struct Material {
  float3 albedo;
  float3 emit;
};
struct Environment {
  float3 sky_color;
};
struct TraversalResult {
  float3 color;
};

SHADER_MAIN
void __closesthit__() {
  const auto& mat = *(const Material*)optixGetSbtDataPointer();
  auto& life = *(RayLife<TraversalResult>*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  life.res.color = mat.albedo;
}

SHADER_MAIN
void __anyhit__() {
}

SHADER_MAIN
void __miss__() {
  const auto& env = *(const Environment*)optixGetSbtDataPointer();
  auto& life = *(RayLife<TraversalResult>*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  life.ttl = 0;
  life.res.color = env.sky_color;
}

SHADER_MAIN
void __raygen__() {
  float4 o = { 0, 0, 0, 1 };
  float4 right = { 1, 0, 0, 0 };
  float4 up = { 0, 1, 0, 0 };
  auto trans = Transform()
    .translate(0, 0, -1)
    .rotate({ 1.0, 0.0, 0.0 }, deg2rad(45.0f))
    .rotate({ 0.0, 1.0, 0.0 }, deg2rad(-45.0f))
    .scale(2, 2, 2);
  o = trans * o;
  right = trans * right;
  up = trans * up;

  Ray ray = ortho_ray(make_float3(o), make_float3(right), make_float3(up));

  RayLife<TraversalResult> life {};
  life.ttl = 1;

  uint32_t wLife[] = PTR2WORDS(&life);
  while (life.ttl--) {
    optixTrace(cfg.trav, ray.o, ray.v,
      0.f, 1e20f, 0.0f, OptixVisibilityMask(255),
      // If you don't use it then YOU SHOULD DISABLE IT to bypass a program
      // invocation.
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,
      0, 1, 0, wLife[0], wLife[1]);
  }

  write_attm(life.res.color);
  //write_attm(color_encode_n1p1(ray.o));
}

}
