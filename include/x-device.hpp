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



#define TRAVERSE(trav, life, ray_flags)                                        \
{                                                                              \
  uint32_t wLife[] = PTR2WORDS(&life);                                         \
  optixTrace(trav, life.ray.o, life.ray.v,                                     \
    1e-5f, 1e20f, 0.0f, OptixVisibilityMask(255),                              \
    ray_flags,                                                                 \
    0, 1, 0, wLife[0], wLife[1]);                                              \
}
#define TRAVERSE_EX(trav, life, tmin, tmax, ray_flags)                         \
{                                                                              \
  uint32_t wLife[] = PTR2WORDS(&life);                                         \
  optixTrace(trav, life.ray.o, life.ray.v,                                     \
    tmin, tmax, 0.0f, OptixVisibilityMask(255),                                \
    ray_flags,                                                                 \
    0, 1, 0, wLife[0], wLife[1]);                                              \
}



//
// ## Sampling Utilities
//

// Standard even-spacing sampling utilities. The following sampling points are
// the referential sampling patterns defined by the Vulkan Specification.
// See Section 24.3. Multisampling of the Vulkan Specification for more
// information.
template<uint32_t TCount>
struct StandardSampler {};
template<>
struct StandardSampler<1> {
  static constexpr const float2 samp_pts[1] = {
    { 0.5, 0.5 },
  };
};
template<>
struct StandardSampler<2> {
  static constexpr const float2 samp_pts[2] = {
    { 0.75, 0.75 },
    { 0.25, 0.25 },
  };
};
template<>
struct StandardSampler<4> {
  static constexpr const float2 samp_pts[4] = {
    { 0.375, 0.125 },
    { 0.875, 0.375 },
    { 0.125, 0.625 },
    { 0.625, 0.875 },
  };
};
template<>
struct StandardSampler<8> {
  static constexpr const float2 samp_pts[8] = {
    { 0.5625, 0.3125 },
    { 0.4375, 0.6875 },
    { 0.8125, 0.5625 },
    { 0.3125, 0.1875 },
    { 0.1875, 0.8125 },
    { 0.0625, 0.4375 },
    { 0.6875, 0.9375 },
    { 0.9375, 0.0625 },
  };
};
template<>
struct StandardSampler<16> {
  static constexpr const float2 samp_pts[16] = {
    { 0.5625, 0.5625 },
    { 0.4375, 0.3125 },
    { 0.3125, 0.625 },
    { 0.75, 0.4375 },
    { 0.1875, 0.375 },
    { 0.625, 0.8125 },
    { 0.8125, 0.6875 },
    { 0.6875, 0.1875 },
    { 0.375, 0.875 },
    { 0.5, 0.0625 },
    { 0.25, 0.125 },
    { 0.125, 0.75 },
    { 0.0, 0.5 },
    { 0.9375, 0.25 },
    { 0.875, 0.9375 },
    { 0.0625, 0.0 },
  };
};





} // namespace liong
