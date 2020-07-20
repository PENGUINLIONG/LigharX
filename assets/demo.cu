#include <cmath>
#include "x.hpp"

namespace liong {

struct Ray {
  float3 o;
  float3 v;
};

struct Material {
  float3 obj_color;
};
struct Environment {
  float3 sky_color;
};


LAUNCH_CFG
LaunchConfig cfg;


SHADER_FN
float2 get_film_coord() {
  uint3 launch_idx = optixGetLaunchIndex();
  auto x = ((float)(launch_idx.x) * 2 + 1 - cfg.launch_size.x) /
    cfg.launch_size.x;
  auto y = ((float)(launch_idx.y) * 2 + 1 - cfg.launch_size.y) /
    cfg.launch_size.y;
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
void write_attm(float3 color) {
  cfg.framebuf[get_invoke_idx()] = color_encode_0p1(color);
}
SHADER_FN
void write_attm(uint32_t color) {
  cfg.framebuf[get_invoke_idx()] = color;
}

/*
// Get a orthogonal projected ray for this raygen shader invocation. The NDC
// coordinates range from -1 to 1.
SHADER_FN
float3 ortho_ray(const Transform& trans) {
  auto ndc_coord();
}
*/

// Get a perspectively projected ray for this raygen shader invocation, from the
// origin of the current coordinate system. The default value is the film
// distance forming 90 degree between the left-most and right-most ray.
SHADER_FN
Ray perspect_ray(float3 o = {}, float film_z = 0.7071f) {
  return Ray { o, normalize(make_float3(get_film_coord(), film_z)) };
}

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
  //write_attm(color_encode_n1p1(make_float3(get_film_coord(), 1.0)));
}

}
