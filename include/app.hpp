#pragma once
#include "common.hpp"

// TODO: (penguinliong) Support multiple types of framebuffer.
struct LaunchConfig {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t* framebuf;
};
