#include <cmath>
#include "x.hpp"
#include "x-device.hpp"

namespace liong {

LAUNCH_CFG
struct LaunchConfig {
  OptixTraversableHandle trav;
  uint32_t* framebuf;
} cfg;

struct Material {
  float3 albedo;
  float3 emit;
};
struct Environment {
  // TODO: (penguinliong) Copy ambient settings as raygen paramters.
  float3 ambient;
};
struct TraversalResult {
  // Point of intersection (hit).
  float3 p;
  // Reflected ray direction.
  float3 refl_v;

  float3 color;
  // Gradually decreasing in magnitude.
  float3 color_mask = { 1, 1, 1 };
};

SHADER_MAIN
void __closesthit__() {
  const auto& mat = GET_SBT(Material);
  auto& life = GET_PAYLOAD(TraversalResult);

  const float F0 = 0.04f;

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

  life.ray.o = p;
  life.ray.v = refl_v;
  life.res.color += mat.emit * life.res.color_mask;
  life.res.color_mask *= F0 * mat.albedo;
}

SHADER_MAIN
void __anyhit__() {
}

SHADER_MAIN
void __miss__() {
  const auto& env = GET_SBT(Environment);
  auto& life = GET_PAYLOAD(TraversalResult);
  life.ttl = 0;
  life.res.color += env.ambient * life.res.color_mask;
}


SHADER_MAIN
void __raygen__() {
  auto launch_prof = get_launch_prof();
  auto trans = Transform()
    .scale(0.5, 0.5, 0.5)
    .rotate({ 0.0, 1.0, 0.0 }, deg2rad(45.0f))
    .rotate({ 1.0, 0.0, 0.0 }, deg2rad(45.0f))
    .translate(0, 0.5, -2)
    .inverse();

  RayLife<TraversalResult> life { ortho_ray(launch_prof, trans), 1, {} };

  uint32_t wLife[] = PTR2WORDS(&life);
  while (life.ttl--) {
    TRAVERSE(cfg.trav, life, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
  }

  cfg.framebuf[launch_prof.invoke_idx] = color_encode_n1p1(life.res.color);
}

}
