#pragma once
// Denoiser extension support for LigharX.
#include "x.hpp"
#include "core.hpp"
#include "ty.hpp"

namespace liong {

struct DenoiserConfig {
  OptixDenoiserInputKind in_kind;
  // Denoiser only work for 2D framebuffers so the third component must only be
  // 1.
  uint3 max_dim;
  PixelFormat fmt;
  // If set to a non-zero value, HDR denoise model is enabled; LDR denoise model
  // is used othersize.
  float hdr_intensity;
};
struct Denoiser {
  OptixDenoiser denoiser;
  DeviceMemory devmem;
  DeviceMemorySlice state_devmem;
  DeviceMemorySlice scratch_devmem;
  DeviceMemorySlice hdr_intensity_devmem;
};



OptixPixelFormat _fmt_lighar2optix(PixelFormat fmt) {
  if (fmt.is_half) {
    switch (fmt.ncomp) {
    case 2: return OPTIX_PIXEL_FORMAT_HALF3;
    case 3: return OPTIX_PIXEL_FORMAT_HALF3;
    }
  } else if (fmt.is_single) {
    switch (fmt.ncomp) {
    case 2: return OPTIX_PIXEL_FORMAT_FLOAT3;
    case 3: return OPTIX_PIXEL_FORMAT_FLOAT3;
    }
  } else if (fmt.int_exp2 == 1) {
    switch (fmt.ncomp) {
    case 2: return OPTIX_PIXEL_FORMAT_UCHAR3;
    case 3: return OPTIX_PIXEL_FORMAT_UCHAR3;
    }
  }
  ASSERT << "False";
}
extern Denoiser create_ldr_denoiser(const Context& ctxt, const DenoiserConfig& cfg) {
  OptixDenoiserOptions denoiser_opt;
  denoiser_opt.inputKind = cfg.in_kind;
  denoiser_opt.pixelFormat = _fmt_lighar2optix(cfg.fmt);
  OptixDenoiser denoiser;
  // FIXME: (penguinliong) Is that all constant zero for floating point number
  // initialy the same representation?
  OptixDenoiserModelKind model_kind = cfg.hdr_intensity != 0 ?
    OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
  OPTIX_ASSERT << optixDenoiserCreate(ctxt.optix_dc, &denoiser_opt, &denoiser);
  OPTIX_ASSERT << optixDenoiserSetModel(denoiser, model_kind, nullptr, 0);
  OptixDenoiserSizes denoiser_size;
  OPTIX_ASSERT << optixDenoiserComputeMemoryResources(denoiser, cfg.max_dim.x,
    cfg.max_dim.y, &denoiser_size);
  size_t hdr_intensity_offset = denoiser_size.stateSizeInBytes +
    denoiser_size.recommendedScratchSizeInBytes;
  size_t alloc_size = hdr_intensity_offset +
    // Size of HDR intensity.
    sizeof(float);
  auto devmem = alloc_mem(alloc_size);
  auto state_devmem = devmem.slice(0, denoiser_size.stateSizeInBytes);
  auto scratch_devmem = devmem.slice(denoiser_size.stateSizeInBytes,
    denoiser_size.recommendedScratchSizeInBytes);
  auto hdr_intensity_devmem = devmem.slice(hdr_intensity_offset, sizeof(float));
  return Denoiser {
    std::move(denoiser),
    std::move(devmem),
    std::move(state_devmem),
    std::move(scratch_devmem),
  };
}
extern void cmd_denoise(
  Transaction& transact,
  const Denoiser& denoiser,
  const Framebuffer& in_framebuf,
  const Framebuffer& out_framebuf
) {
  ASSERT << ((in_framebuf.dim.z == 1) && (out_framebuf.dim.z == 1))
    << "framebuffer whose z-dimension is not 1 cannot be denoised";
  ASSERT << (in_framebuf.dim == out_framebuf.dim)
    << "framebuffer size mismatched";

  OPTIX_ASSERT << optixDenoiserSetup(denoiser.denoiser, transact.stream,
    in_framebuf.dim.x, in_framebuf.dim.y,
    denoiser.state_devmem.ptr, denoiser.state_devmem.size,
    denoiser.scratch_devmem.ptr, denoiser.scratch_devmem.size);

  OptixDenoiserParams params;
  params.denoiseAlpha = false;
  params.blendFactor = 0;
  params.hdrIntensity = denoiser.hdr_intensity_devmem.ptr;
  OptixImage2D in_img {};
  in_img.data = in_framebuf.framebuf_devmem.ptr;
  in_img.format = _fmt_lighar2optix(in_framebuf.fmt);
  in_img.width = in_framebuf.dim.x;
  in_img.height = in_framebuf.dim.y;
  in_img.pixelStrideInBytes = in_framebuf.fmt.get_fmt_size();
  in_img.rowStrideInBytes = in_img.pixelStrideInBytes * in_img.width;
  OptixImage2D out_img {};
  out_img.data = out_framebuf.framebuf_devmem.ptr;
  out_img.format = _fmt_lighar2optix(out_framebuf.fmt);
  out_img.width = out_framebuf.dim.x;
  out_img.height = out_framebuf.dim.y;
  out_img.pixelStrideInBytes = out_framebuf.fmt.get_fmt_size();
  out_img.rowStrideInBytes = out_img.pixelStrideInBytes * in_img.width;
  OPTIX_ASSERT << optixDenoiserInvoke(denoiser.denoiser, transact.stream,
    &params, denoiser.state_devmem.ptr, denoiser.state_devmem.size,
    &in_img, 1, 0, 0,
    &out_img, denoiser.scratch_devmem.ptr, denoiser.scratch_devmem.size);
}

} // namespace liong
