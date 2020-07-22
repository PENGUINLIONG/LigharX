#include <cmath>
#include "x.hpp"
#include "x-device.hpp"

namespace liong {

// Everything valuable during the lifetime of a ray.
template<typename TRes>
struct RayLife {
  // Point of intersection (hit).
  float3 p;
  // Reflected ray direction.
  float3 refl_v;
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
  // TODO: (penguinliong) Copy ambient settings as raygen paramters.
  float3 ambient;
};
struct TraversalResult {
  float3 color;
  // Gradually decreasing in magnitude.
  float3 color_mask = { 1, 1, 1 };
};

SHADER_MAIN
void __closesthit__() {
  const auto& mat = *(const Material*)optixGetSbtDataPointer();

  const float F0 = 0.04f;

  uint32_t wLife[] = { optixGetPayload_0(), optixGetPayload_1() };
  auto& life = *(RayLife<TraversalResult>*)WORDS2PTR(wLife[0], wLife[1]);

  constexpr uint32_t sbt_idx = 0;
  auto gas_trav = optixGetGASTraversableHandle();
  float3 tri[3];
  optixGetTriangleVertexData(gas_trav, optixGetPrimitiveIndex(),
    sbt_idx, 0.f, tri);

  // Calculate the point of hit.
  float2 bary = optixGetTriangleBarycentrics();
  auto norm = normalize(cross(tri[2] - tri[0], tri[1] - tri[0]));
  norm = optixTransformNormalFromObjectToWorldSpace(norm);
  auto refl_v = -reflect(optixGetWorldRayDirection(), norm);
  refl_v = normalize(refl_v);

  auto p = bary.x * tri[0] + bary.y * tri[1] +
    (1 - (bary.x + bary.y)) * tri[2];
  p = optixTransformPointFromObjectToWorldSpace(p);

  life.refl_v = refl_v;
  life.p = p;
  life.res.color += mat.emit * life.res.color_mask;
  life.res.color_mask *= F0 * mat.albedo;
}

SHADER_MAIN
void __anyhit__() {
}

SHADER_MAIN
void __miss__() {
  const auto& env = *(const Environment*)optixGetSbtDataPointer();
  auto& life = *(RayLife<TraversalResult>*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1());
  life.ttl = 0;
  life.res.color += env.ambient * life.res.color_mask;
}


SHADER_MAIN
void __raygen__() {
  auto trans = Transform()
    .scale(0.5, 0.5, 0.5)
    .rotate({ 0.0, 1.0, 0.0 }, deg2rad(45.0f))
    .rotate({ 1.0, 0.0, 0.0 }, deg2rad(45.0f))
    .translate(0, 0.5, -2)
    .inverse();

  float3 color;

  Ray ray = ortho_ray(trans);
  RayLife<TraversalResult> life {};
  life.ttl = 1;

  uint32_t wLife[] = PTR2WORDS(&life);
  while (life.ttl--) {
    optixTrace(cfg.trav, ray.o, ray.v,
      1e-5f, 1e20f, 0.0f, OptixVisibilityMask(255),
      // If you don't use it then YOU SHOULD DISABLE IT to bypass a program
      // invocation.
      OPTIX_RAY_FLAG_DISABLE_ANYHIT,
      0, 1, 0, wLife[0], wLife[1]);
    ray.o = life.p;
    ray.v = life.refl_v;
  }

  write_attm(life.res.color);
  //write_attm(make_float3(get_film_coord()));
  //write_attm(color_encode_n1p1(life.res.color));
}

}
