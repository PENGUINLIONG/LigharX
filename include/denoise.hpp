#pragma once
// Denoiser extension support for LigharX.
#include "x.hpp"
#include "core.hpp"
#include "ty.hpp"

namespace liong {

namespace denoise {

struct DenoiserConfig {
  OptixDenoiserInputKind in_kind;
  // Denoiser only work for 2D framebuffers so the third component must only be
  // 1.
  uint3 max_dim;
  PixelFormat fmt;
  // If set to a non-zero value, HDR denoise model is enabled; LDR denoise model
  // is used othersize.
  float hdr_intensity;
  // Allow the framebuffer to be the input and output at the same time.
  bool inplace;
};
struct Denoiser {
  OptixDenoiser denoiser;
  DeviceMemory devmem;
  DeviceMemorySlice state_devmem;
  DeviceMemorySlice scratch_devmem;
  DeviceMemorySlice hdr_intensity_devmem;
};



extern Denoiser create_denoiser(const Context& ctxt, const DenoiserConfig& cfg);
extern void destroy_denoiser(Denoiser& denoiser);


extern void cmd_denoise(
  Transaction& transact,
  const Denoiser& denoiser,
  const Framebuffer& in_framebuf,
  const Framebuffer& out_framebuf,
  // Denoise the alpha channel or just copy the original alpha value?
  bool denoise_alpha = false
);

} // namespace denoise

} // namespace liong
