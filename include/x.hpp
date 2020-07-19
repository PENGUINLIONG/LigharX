#pragma once
// Host/device cross-end definition header.
// @PENGUINLIONG
#include <cstdint>
#include <cmath>
#include <optix_stubs.h>
#ifdef __CUDACC__
#include <optix_device.h>
// Cross-platform compilable function.
#define X __device__
#else
#include <optix.h>
#define X
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
constexpr uint32_t pack_unorm4_abgr(float4 x) {
  return ((uint32_t)(x.x * 255)) |
    ((uint32_t)(x.y * 255) << 8) |
    ((uint32_t)(x.z * 255) << 16) |
    ((uint32_t)(x.w * 255) << 24);
}
constexpr uint32_t pack_unorm3_abgr(float3 x) {
  return ((uint32_t)(x.x * 255)) |
    ((uint32_t)(x.y * 255) << 8) |
    ((uint32_t)(x.z * 255) << 16) | 0xFF000000;
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
  inline X float3 operator*(const float3& rhs) const {
    return float3 {
      dot(*((float3*)&rows[0]), rhs),
      dot(*((float3*)&rows[1]), rhs),
      dot(*((float3*)&rows[2]), rhs)
    };
  }
  inline X Transform operator*(const Transform& rhs) const {
    return Transform {
      mat[0] * rhs.mat[0] + mat[1] * rhs.mat[4] + mat[2] * rhs.mat[8],
      mat[0] * rhs.mat[1] + mat[1] * rhs.mat[5] + mat[2] * rhs.mat[9],
      mat[0] * rhs.mat[2] + mat[1] * rhs.mat[6] + mat[2] * rhs.mat[10],
      mat[0] * rhs.mat[3] + mat[1] * rhs.mat[7] + mat[2] * rhs.mat[11],

      mat[4] * rhs.mat[0] + mat[5] * rhs.mat[4] + mat[6] * rhs.mat[8],
      mat[4] * rhs.mat[1] + mat[5] * rhs.mat[5] + mat[6] * rhs.mat[9],
      mat[4] * rhs.mat[2] + mat[5] * rhs.mat[6] + mat[6] * rhs.mat[10],
      mat[4] * rhs.mat[3] + mat[5] * rhs.mat[7] + mat[6] * rhs.mat[11],

      mat[8] * rhs.mat[0] + mat[9] * rhs.mat[4] + mat[10] * rhs.mat[8],
      mat[8] * rhs.mat[1] + mat[9] * rhs.mat[5] + mat[10] * rhs.mat[9],
      mat[8] * rhs.mat[2] + mat[9] * rhs.mat[6] + mat[10] * rhs.mat[10],
      mat[8] * rhs.mat[3] + mat[9] * rhs.mat[7] + mat[10] * rhs.mat[11],
    };
  }
  inline X Transform scale(float x, float y, float z) const {
    return Transform { x,0,0,0, 0,y,0,0, 0,0,z,0 } *(*this);
  }
  inline X Transform translate(float x, float y, float z) const {
    return Transform { 0,0,0,x, 0,0,0,y, 0,0,0,z } *(*this);
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



// TODO: (penguinliong) Support multiple types of framebuffer.
template<typename T = CUdeviceptr>
struct LaunchConfig {
  template<typename TMat>
  using ptrsel = std::conditional_t<std::is_same_v<TMat, CUdeviceptr>, CUdeviceptr, const TMat*>;

  uint3 launch_size;
  OptixTraversableHandle trav;
  typename ptrsel<T> mat;
  uint32_t* framebuf;
};

} // namespace liong
