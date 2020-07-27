#pragma once
#ifndef __CUDACC__
static_assert(false, "cannot include device-only header file in host code.");
#endif
// Device-only utility subroutines.
// @PENGUINLIONG
#include <x.hpp>

namespace liong {

//
// ## Predefined Launch Configuration
//
// Because the configuration is DEFINED here in the header file, you have to
// write all the stages in the same `.cu` source, otherwise you would get some
// problem linking all the things up.
//

LAUNCH_CFG
LaunchConfig cfg;



//
// ## Predefined Types
//

struct Ray {
  float3 o;
  float3 v;
};
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

SHADER_FN
float2 get_film_coord_n1p1() {
  uint3 launch_idx = optixGetLaunchIndex();
  auto x = ((float)(launch_idx.x) * 2 + 1 - cfg.launch_size.x) /
    cfg.launch_size.x;
  auto y = ((float)(launch_idx.y) * 2 + 1 - cfg.launch_size.y) /
    cfg.launch_size.y;
  auto rel_pos = make_float2(x, y);
  return rel_pos;
}
SHADER_FN
float2 get_film_coord_0p1() {
  uint3 launch_idx = optixGetLaunchIndex();
  auto x = ((float)(launch_idx.x) * 2 + 1) /
    (2 * cfg.launch_size.x);
  auto y = ((float)(launch_idx.y) * 2 + 1) /
    (2 * cfg.launch_size.y);
  auto rel_pos = make_float2(x, y);
  return rel_pos;
}
SHADER_FN
uint32_t get_invoke_idx() {
  uint3 launch_idx = optixGetLaunchIndex();
  return (launch_idx.z * cfg.launch_size.y + launch_idx.y) * cfg.launch_size.x +
      launch_idx.x;
}
SHADER_FN
void write_attm_n1p1(float3 color) {
  cfg.framebuf[get_invoke_idx()] = color_encode_n1p1(color);
}
SHADER_FN
void write_attm_0p1(float3 color) {
  cfg.framebuf[get_invoke_idx()] = color_encode_0p1(color);
}
SHADER_FN
void write_attm(uint32_t color) {
  cfg.framebuf[get_invoke_idx()] = color;
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
  const Transform& trans
) {
  auto cam_coord = make_cam_coord(trans);
  // Let the rays shoot into the screen.
  float3 front = normalize(cross(cam_coord.up, cam_coord.right));
  float2 uv = get_film_coord_n1p1();
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
  const Transform& trans,
  // By default we look at objects from positive-Z to negative-Z in RHS.
  float film_z = 0.7071f
) {
  auto cam_coord = make_cam_coord(trans);
  float3 front = normalize(cross(cam_coord.up, cam_coord.right));
  float2 uv = get_film_coord_n1p1();
  float3 v = normalize(uv.x * cam_coord.right + uv.y * cam_coord.up +
    film_z * front);
  return Ray { cam_coord.o, v };
}



#define TRAVERSE(trav, life, ray_flags)                                        \
{                                                                              \
uint32_t wLife[] = PTR2WORDS(&life);                                           \
optixTrace(trav, life.ray.o, life.ray.v,                                       \
  1e-5f, 1e20f, 0.0f, OptixVisibilityMask(255),                                \
  ray_flags,                                                                   \
  0, 1, 0, wLife[0], wLife[1]);                                                \
}



//
// ## Sampling Utilities
//

// Standard even-spacing sampling utilities.
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
