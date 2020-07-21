#pragma once
#ifndef __CUDACC__
static_assert(false, "cannot include device-only header file in host code.");
#endif
// Device-only utility subroutines.
// @PENGUINLIONG
#include <x.hpp>

namespace liong {

//
// # Predefined Launch Configuration
//
// Because the configuration is DEFINED here in the header file, you have to
// write all the stages in the same `.cu` source, otherwise you would get some
// problem linking all the things up.
//

LAUNCH_CFG
LaunchConfig cfg;



//
// # Predefined Types
//

struct Ray {
	float3 o;
	float3 v;
};



//
// # Raygen and Scheduling Utilities
//

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

// Get a orthogonally projected ray for this raygen shader invocation.
SHADER_FN
Ray ortho_ray(
    float3 o = {},
    float3 right = { 1, 0, 0 },
    float3 up = { 0, 1, 0 }
) {
    float3 front = normalize(cross(right, up));
    float2 uv = get_film_coord();
    o += uv.x * right + uv.y * up;
    return Ray { o, front };
}

// Get a perspectively projected ray for this raygen shader invocation, from the
// origin of the current coordinate system. The default value is the film
// distance forming 90 degree between the left-most and right-most ray.
//
// NOTE: The `right` and `up` parameters' magnitude CAN be used to set up aspect
// ratios.
SHADER_FN
Ray perspect_ray(
    float3 o = {},
    float3 right = { 1, 0, 0 },
    float3 up = { 0, 1, 0 },
    // By default we look at objects from positive-Z to negative-Z in RHS.
    float film_z = 0.7071f
) {
    float3 front = normalize(cross(right, up));
    float2 uv = get_film_coord();
    float3 v = normalize(uv.x * right + uv.y * up + film_z * front);
    return Ray { o, v };
}







} // namespace liong