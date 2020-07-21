#pragma once
// Host/device cross-end definition header.
// @PENGUINLIONG
#include <cstdint>
#include <cmath>
#include <optix_stubs.h>
#ifdef __CUDACC__
#include <optix_device.h>
// Cross-platform compilable function.
#define X __forceinline__ __device__
#else
#include <optix.h>
#define X inline
#endif // __CUDACC__
#include "vector_math.h"

// CUDA-only utility macros.
#ifdef __CUDACC__

#define LAUNCH_CFG extern "C" __constant__
#define SHADER_MAIN extern "C" __global__ 
#define SHADER_FN __forceinline__ __device__

#define PTR2WORDS(ptr) {                                                       \
  (uint32_t)(((uint64_t)(ptr)) >> 32),                                         \
  (uint32_t)(((uint64_t)(ptr)) & 0xFFFFFFFF)                                   \
}
#define WORDS2PTR(w1, w0) \
((const void*)((((uint64_t)(w1)) << 32) | ((uint64_t)(w0))))

#endif // __CUDACC__


namespace liong {

constexpr float4 make_pt(float x, float y, float z) {
  return float4{ x, y, z, 1.0f };
}
constexpr float4 make_vec(float x, float y, float z) {
  return float4{ x, y, z, 0.0f };
}
X uint32_t pack_unorm4_abgr(float4 x) {
  x = clamp(x, 0, 1);
  return ((uint32_t)(x.x * 255.999)) |
    ((uint32_t)(x.y * 255.999) << 8) |
    ((uint32_t)(x.z * 255.999) << 16) |
    ((uint32_t)(x.w * 255.999) << 24);
}
X uint32_t pack_unorm3_abgr(float3 x) {
  x = clamp(x, 0, 1);
  return ((uint32_t)(x.x * 255.999)) |
    ((uint32_t)(x.y * 255.999) << 8) |
    ((uint32_t)(x.z * 255.999) << 16) | 0xFF000000;
}
// Color encode real numbers in [0.0, 1.0] to integers in [0, 255] with linear
// spacing, i.e., The chance a number rounded to 0 is equal to that rounded to
// 255.
X uint32_t color_encode_0p1(const float3& x) {
  return pack_unorm3_abgr(x);
}
// Color encode real numbers in [-1.0, 1.0] to integers in [0, 255] with linear
// spacing, i.e., The chance a number rounded to 0 is equal to that rounded to
// 255.
X uint32_t color_encode_n1p1(const float3& x) {
  return color_encode_0p1(x / 2 + 0.5f);
}
X float3 unquant_unorm8_rgb(uint8_t r, uint8_t g, uint8_t b) {
  return make_float3(r / 255.0f, g / 255.0f, b / 255.0f);
}

constexpr float deg2rad(float deg) {
  return deg / 180.0f * M_PI;
}
constexpr float rad2deg(float deg) {
  return deg / M_PI * 180.0f;
}

// An 3x4 (float32) matrix transform.
struct Transform {
  union {
    float mat[12];
    float4 rows[3];
  };

  X Transform() : mat { 1,0,0,0, 0,1,0,0, 0,0,1,0 } {}
  X Transform(float4 r1, float4 r2, float4 r3) : rows{ r1, r2, r3 } {}
  X Transform(float a, float b, float c, float d,
    float e, float f, float g, float h,
    float i, float j, float k, float l) : mat { a,b,c,d,e,f,g,h,i,j,k,l } {}
  X Transform(float mat[12]) { mat = mat; }
  X Transform(const Transform&) = default;
  X Transform(Transform&&) = default;
  X Transform& operator=(const Transform&) = default;
  X Transform& operator=(Transform&&) = default;

  inline X float4 operator*(const float4& rhs) const {
    return float4 {
      dot(rows[0], rhs), dot(rows[1], rhs), dot(rows[2], rhs), rhs.w
    };
  }
  inline X Transform operator*(const Transform& rhs) const {
    return Transform {
      mat[0] * rhs.mat[0] + mat[1] * rhs.mat[4] + mat[2] * rhs.mat[8],
      mat[0] * rhs.mat[1] + mat[1] * rhs.mat[5] + mat[2] * rhs.mat[9],
      mat[0] * rhs.mat[2] + mat[1] * rhs.mat[6] + mat[2] * rhs.mat[10],
      mat[0] * rhs.mat[3] + mat[1] * rhs.mat[7] + mat[2] * rhs.mat[11]
        + mat[3],

      mat[4] * rhs.mat[0] + mat[5] * rhs.mat[4] + mat[6] * rhs.mat[8],
      mat[4] * rhs.mat[1] + mat[5] * rhs.mat[5] + mat[6] * rhs.mat[9],
      mat[4] * rhs.mat[2] + mat[5] * rhs.mat[6] + mat[6] * rhs.mat[10],
      mat[4] * rhs.mat[3] + mat[5] * rhs.mat[7] + mat[6] * rhs.mat[11]
        + mat[7],

      mat[8] * rhs.mat[0] + mat[9] * rhs.mat[4] + mat[10] * rhs.mat[8],
      mat[8] * rhs.mat[1] + mat[9] * rhs.mat[5] + mat[10] * rhs.mat[9],
      mat[8] * rhs.mat[2] + mat[9] * rhs.mat[6] + mat[10] * rhs.mat[10],
      mat[8] * rhs.mat[3] + mat[9] * rhs.mat[7] + mat[10] * rhs.mat[11]
        + mat[11],
    };
  }
  inline X Transform scale(float x, float y, float z) const {
    return Transform { x,0,0,0, 0,y,0,0, 0,0,z,0 } *(*this);
  }
  inline X Transform scale(float3 v) const {
    return scale(v.x, v.y, v.z);
  }
  inline X Transform translate(float x, float y, float z) const {
    return Transform { 1,0,0,x, 0,1,0,y, 0,0,1,z } *(*this);
  }
  inline X Transform translate(float3 v) const {
    return translate(v.x, v.y, v.z);
  }
  inline X Transform rotate(float x, float y, float z, float rad) const {
    auto sin = std::sinf(rad);
    auto cos = std::cosf(rad);
    auto rcos = 1.0f - cos;
    return Transform {
        cos + rcos * x * x, rcos * x * y - sin * z, rcos * x * z + sin * y, 0,
        rcos * y * x + sin * z, cos + rcos * y * y, rcos * y * z - sin * x, 0,
        rcos * z * x - sin * y, rcos * z * y + sin * x, cos + rcos * z * z, 0,
    } *(*this);
  }
  inline X Transform rotate(float3 axis, float rad) const {
    return rotate(axis.x, axis.y, axis.z, rad);
  }
  inline X Transform inverse() const {
    auto det = mat[0] * mat[5] * mat[10] -
      mat[0] * mat[6] * mat[9] -
      mat[4] * mat[1] * mat[10] +
      mat[4] * mat[2] * mat[9] +
      mat[8] * mat[1] * mat[6] -
      mat[8] * mat[2] * mat[5];
    return Transform {
      (mat[5] * mat[10] - mat[9] * mat[6]) / det,
      (mat[2] * mat[9] - mat[1] * mat[10]) / det,
      (mat[1] * mat[10] - mat[2] * mat[5]) / det,
      -mat[3],

      (mat[6] * mat[8] - mat[4] * mat[10]) / det,
      (mat[0] * mat[10] - mat[2] * mat[8]) / det,
      (mat[4] * mat[6] - mat[0] * mat[6]) / det,
      -mat[7],

      (mat[4] * mat[9] - mat[5] * mat[8]) / det,
      (mat[1] * mat[8] - mat[0] * mat[9]) / det,
      (mat[0] * mat[5] - mat[4] * mat[1]) / det,
      -mat[11]
    };
  }
};
static_assert(sizeof(Transform) == sizeof(float[12]),
  "type `Transform` have incompatable size than optix transform");



// TODO: (penguinliong) Support multiple types of framebuffer.
struct LaunchConfig {
  uint3 launch_size;
  OptixTraversableHandle trav;
  uint32_t* framebuf;
};

} // namespace liong
