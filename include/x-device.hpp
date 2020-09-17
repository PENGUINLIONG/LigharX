#pragma once
#ifndef __CUDACC__
static_assert(false, "cannot include device-only header file in host code.");
#endif
// Device-only utility subroutines.
// @PENGUINLIONG
#include <x.hpp>

namespace liong {

//
// ## Predefined Types
//

// Everything valuable during the lifetime of a ray.
template<typename TRes>
struct RayLife {
  Ray ray;
  // Time-to-live. The number of remaining chances to launch traversal without
  // blasting the stack memory. It SHOULD be decresed before another traversal
  // and SHOULD be set to 0 on a miss.
  uint32_t ttl;
  // User-defined result data.
  TRes res;
};



//
// ## Material Access
//

#define GET_SBT(T)                                                             \
  (*(T*)optixGetSbtDataPointer())
#define GET_PAYLOAD(TRes)                                                      \
  (*(RayLife<TRes>*)WORDS2PTR(optixGetPayload_0(), optixGetPayload_1()))

//
// ## Raygen and Scheduling Utilities
//

struct LaunchProfile {
  uint3 launch_size;
  uint3 launch_idx;
  uint32_t invoke_idx;
};
SHADER_FN
LaunchProfile get_launch_prof() {
  uint3 launch_size = optixGetLaunchDimensions();
  uint3 launch_idx = optixGetLaunchIndex();
  uint32_t invoke_idx = (launch_idx.z * launch_size.y + launch_idx.y) * launch_size.x +
    launch_idx.x;
  return LaunchProfile {
    std::move(launch_size),
    std::move(launch_idx),
    std::move(invoke_idx),
  };
}
SHADER_FN
float2 get_film_coord_n1p1(const LaunchProfile& launch_prof) {
  const uint3& launch_size = launch_prof.launch_size;
  const uint3& launch_idx = launch_prof.launch_idx;
  auto x = ((float)(launch_idx.x) * 2 + 1 - launch_size.x) / launch_size.x;
  auto y = ((float)(launch_idx.y) * 2 + 1 - launch_size.y) / launch_size.y;
  auto rel_pos = make_float2(x, y);
  return rel_pos;
}
SHADER_FN
float2 get_film_coord_0p1(const LaunchProfile& launch_prof) {
  const uint3& launch_size = launch_prof.launch_size;
  const uint3& launch_idx = launch_prof.launch_idx;
  auto x = ((float)(launch_idx.x) * 2 + 1) / (2 * launch_size.x);
  auto y = ((float)(launch_idx.y) * 2 + 1) / (2 * launch_size.y);
  auto rel_pos = make_float2(x, y);
  return rel_pos;
}
struct CameraCoordinates {
  float3 o;
  float3 right;
  float3 up;
};
SHADER_FN
CameraCoordinates make_cam_coord(const Transform& trans) {
  auto o = trans * float4 { 0, 0, 0, 1 };
  auto right = trans * float4 { 1, 0, 0, 0 };
  auto up = trans * float4 { 0, 1, 0, 0 };
  return { make_float3(o), make_float3(right), make_float3(up) };
}
// Get a orthogonally projected ray for this raygen shader invocation.
SHADER_FN
Ray ortho_ray(
  const LaunchProfile& launch_prof,
  const Transform& trans
) {
  auto cam_coord = make_cam_coord(trans);
  // Let the rays shoot into the screen.
  float3 front = normalize(cross(cam_coord.up, cam_coord.right));
  float2 uv = get_film_coord_n1p1(launch_prof);
  cam_coord.o += uv.x * cam_coord.right + uv.y * cam_coord.up;
  return Ray { cam_coord.o, front };
}

// Get a perspectively projected ray for this raygen shader invocation, from the
// origin of the current coordinate system. The default value is the film
// distance forming 90 degree between the left-most and right-most ray.
//
// NOTE: The `right` and `up` parameters' magnitude CAN be used to set up aspect
// ratios.
SHADER_FN
Ray perspect_ray(
  const LaunchProfile& launch_prof,
  const Transform& trans,
  // By default we look at objects from positive-Z to negative-Z in RHS.
  float film_z = 0.7071f
) {
  auto cam_coord = make_cam_coord(trans);
  float3 front = normalize(cross(cam_coord.up, cam_coord.right));
  float2 uv = get_film_coord_n1p1(launch_prof);
  float3 v = normalize(uv.x * cam_coord.right + uv.y * cam_coord.up +
    film_z * front);
  return Ray { cam_coord.o, v };
}



#define TRAVERSE_GROUP_EX(trav, igrp, ngrp, life, tmin, tmax, ray_flags)       \
{                                                                              \
  uint32_t wLife[] = PTR2WORDS(&life);                                         \
  optixTrace(trav, life.ray.o, life.ray.v,                                     \
    tmin, tmax, 0.0f, OptixVisibilityMask(255),                                \
    ray_flags,                                                                 \
    igrp, ngrp, igrp, wLife[0], wLife[1]);                                        \
}
#define TRAVERSE_GROUP(trav, igrp, ngrp, life, ray_flags)                      \
  TRAVERSE_GROUP_EX(trav, igrp, ngrp, life, 1e-5, 1e20, ray_flags)
#define TRAVERSE_EX(trav, life, tmin, tmax, ray_flags)                         \
  TRAVERSE_GROUP_EX(trav, 0, 1, life, tmin, tmax, ray_flags)
#define TRAVERSE(trav, life, ray_flags)                                        \
  TRAVERSE_GROUP(trav, 0, 1, life, ray_flags)

#define INTERPOLATE(buf, prim, bary)                                           \
  buf[prim.y] * bary.x +                                                       \
  buf[prim.z] * bary.y +                                                       \
  buf[prim.x] * (1 - (bary.x + bary.y))


} // namespace liong
