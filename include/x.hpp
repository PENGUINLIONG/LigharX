#pragma once
// Host/device cross-end definition header.
// @PENGUINLIONG

#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <optix_stubs.h>
#ifdef __CUDACC__
#include <cuda.h>
#include <optix_device.h>
// Cross-platform compilable function.
#define X __device__
#else
#include <cuda_runtime_api.h>
#include <optix.h>
#define X
#endif

namespace liong {

// TODO: (penguinliong) Support multiple types of framebuffer.
struct LaunchConfig {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  OptixTraversableHandle trav;
  uint32_t* framebuf;
};

#define PTR2WORDS(ptr) {                                                       \
  (uint32_t)(((uint64_t)(ptr)) >> 32),                                         \
  (uint32_t)(((uint64_t)(ptr)) & 0xFFFFFFFF)                                   \
}
#define WORDS2PTR(w1, w0) ((const void*)((((uint64_t)(w1)) << 32) | ((uint64_t)(w0))))


template<typename T, size_t TSize,
  typename _ = std::enable_if<std::is_arithmetic_v<T> && TSize <= 4>>
struct Vector {
  T data[TSize];
  Vector() : data {} {}
  Vector(const Vector<T, TSize>& b) {
    for (auto i = 0; i < TSize; ++i) {
      data[i] = b[i];
    }
  }
  Vector(Vector<T, TSize>&& b) {
    for (auto i = 0; i < TSize; ++i) {
      data[i] = b[i];
    }
  }
  Vector(const std::initializer_list<T>& b) {
    for (auto i = 0; i < TSize; ++i) {
      data[i] = b[i];
    }
  }
  Vector<T, TSize>& operator=(const Vector<T, TSize>& b) {
    for (auto i = 0; i < TSize; ++i) {
      data[i] = b[i];
    }
  }
  Vector<T, TSize>& operator=(Vector<T, TSize>&& b) {
    for (auto i = 0; i < TSize; ++i) {
      data[i] = b[i];
    }
  }

#define DEF_BIN_OP(x)                                                          \
  constexpr Vector<T, TSize> operator x(const Vector<T, TSize>& rhs) const {   \
    Vector<T, TSize> rv {};                                                    \
    for (auto i = 0; i < TSize; ++i) {                                         \
      rv.data[i] = data[i] x rhs.data[i];                                      \
    }                                                                          \
    return rv;                                                                 \
  }
  DEF_BIN_OP(+);
  DEF_BIN_OP(-);
  DEF_BIN_OP(*);
  DEF_BIN_OP(/ );
#undef DEF_BIN_OP

  template<typename _ = std::enable_if_t<std::is_floating_point_v<T>>>
  constexpr uint32_t to_unorm_pack() const {
    uint32_t rv {};
    for (auto i = 0; i < TSize; ++i) {
      rv |= ((uint32_t)(data[i] * 255.999) << (2 * i));
    }
    return rv;
  }
  constexpr T dot(const Vector<T, TSize>& rhs) const {
    T rv {};
    for (auto i = 0; i < TSize; ++i) {
      rv += data[i] * rhs.data[i];
    }
    return rv;
  }
  constexpr T mag() const {
    return (T)std::sqrt(dot(*this));
  }
  constexpr Vector<T, TSize> normalize() const {
    Vector<T, TSize> rv;
    auto m = mag();
    for (auto i = 0; i < TSize; ++i) {
      rv[i] = data[i] / m;
    }
    return rv;
  }

};

using vec2 = Vector<float, 2>;
using vec3 = Vector<float, 3>;
using vec4 = Vector<float, 4>;
using ivec2 = Vector<int32_t, 2>;
using ivec3 = Vector<int32_t, 3>;
using ivec4 = Vector<int32_t, 4>;


// An 3x4 matrix transform.
template<typename T,
  typename _ = std::enable_if_t<std::is_floating_point_v<T>>>
  struct Transform {
  T mat[12];

  X Transform() : mat { 1,0,0,0, 0,1,0,0, 0,0,1,0 } {}
  X Transform(T a, T b, T c, T d,
    T e, T f, T g, T h,
    T i, T j, T k, T l) : mat { a,b,c,d,e,f,g,h,i,j,k,l } {}
  X Transform(T mat[12]) { mat = mat; }
  X Transform(const Transform&) = default;
  X Transform(Transform&&) = default;
  X Transform& operator=(const Transform&) = default;
  X Transform& operator=(Transform&&) = default;

  template<typename T2, size_t TSize,
    typename _ = std::enable_if_t<std::is_floating_point_v<T2> && TSize <= 4>>
  constexpr X Vector<T2, TSize> operator*(const Vector<T2, TSize>& rhs) const {
    Vector<T2, 3> rv {};
    for (auto i = 0; i < TSize; ++i) {
      for (auto j = 0; j < TSize; ++j) {
        rv[i] += data[i * 4 + j] * rhs.data[j];
      }
    }
    return rv;
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
  inline X Transform scale(T x, T y, T z) const {
    return Transform { x,0,0,0, 0,y,0,0, 0,0,z,0 } *(*this);
  }
  inline X Transform translate(T x, T y, T z) const {
    return Transform { 0,0,0,x, 0,0,0,y, 0,0,0,z } *(*this);
  }
  inline X Transform rotate(T x, T y, T z, T rad) const {
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

using mat3x4 = Transform<float>;

}
