#include "x.hpp"

namespace liong {

struct Material {
  float3 obj_color;
};
struct Environment {
  float3 sky_color;
};


LAUNCH_CFG
LaunchConfig cfg;


SHADER_FN
float3 get_film_coord() {
  uint3 launch_idx = optixGetLaunchIndex();
  auto rel_pos = make_float3(
    (((float)launch_idx.x) * 2 + 1) / (cfg.launch_size.x * 2),
    (((float)launch_idx.y) * 2 + 1) / (cfg.launch_size.y * 2),
    (((float)launch_idx.z) * 2 + 1) / (cfg.launch_size.z * 2)
  );
  return rel_pos;
}
SHADER_FN
uint32_t get_invoke_idx() {
  uint3 launch_idx = optixGetLaunchIndex();
  return (launch_idx.z * cfg.launch_size.y + launch_idx.x) * cfg.launch_size.x +
    launch_idx.x;
}

/*
// Get a orthogonal projected ray for this raygen shader invocation. The NDC
// coordinates range from -1 to 1.
SHADER_FN
float3 ortho_ray(const Transform& trans) {
  auto ndc_coord();
}
// Get a perspective projected ray for this raygen shader invocation. the NDC
// coordinates range from -1
float3 perspect_ray() {
  get_film_coord()
}
*/


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
  float3 film_coord = get_film_coord();
  film_coord.y = 1;
  float3 color{};
  uint32_t wColor[] = PTR2WORDS(&color);

  optixTrace(cfg.trav, film_coord, {0.0, 0.0, -1.0},
    0.f, 1e20f, 0.0f, OptixVisibilityMask(255),
    // If you don't use it then YOU SHOULD DISABLE IT to bypass a program
    // invocation.
    OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    0, 1, 0, wColor[0], wColor[1]);

  uint32_t i = get_invoke_idx();
  cfg.framebuf[i] = pack_unorm3_abgr(color);
}

}
