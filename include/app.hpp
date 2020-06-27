#pragma once
#include "common.hpp"
#include <optix_types.h>

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
#define WORDS2PTR(w1, w0) ((void*)((((uint64_t)(w1)) << 32) | ((uint64_t)(w0))))
