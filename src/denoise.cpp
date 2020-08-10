#include "denoise.hpp"
#include "log.hpp"

namespace liong {

namespace denoise {

OptixPixelFormat _fmt_lighar2optix(PixelFormat fmt) {
  if (fmt.is_half) {
    switch (fmt.ncomp) {
    case 2: return OPTIX_PIXEL_FORMAT_HALF3;
    case 3: return OPTIX_PIXEL_FORMAT_HALF4;
    }
  } else if (fmt.is_single) {
    switch (fmt.ncomp) {
    case 2: return OPTIX_PIXEL_FORMAT_FLOAT3;
    case 3: return OPTIX_PIXEL_FORMAT_FLOAT4;
    }
  } else if (fmt.int_exp2 != 0) {
    ASSERT << false
      << "quantized data cannot be denoised";
  }
  ASSERT << false
    << "format not supported by optix";
}

Denoiser create_denoiser(const Context& ctxt, const DenoiserConfig& cfg) {
  OptixDenoiserOptions denoiser_opt;
  denoiser_opt.inputKind = cfg.in_kind;
#if OPTIX_VERSION < 70100
  denoiser_opt.pixelFormat = _fmt_lighar2optix(cfg.fmt);
#endif // OPTIX_VERSION < 70100
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
#if OPTIX_VERSION < 70100
  size_t scratch_size = denoiser_size.recommendedScratchSizeInBytes;
#else
  size_t scratch_size = cfg.inplace ?
    denoiser_size.withOverlapScratchSizeInBytes :
    denoiser_size.withoutOverlapScratchSizeInBytes;
#endif
  size_t hdr_intensity_offset = denoiser_size.stateSizeInBytes + scratch_size;

  size_t alloc_size = hdr_intensity_offset +
    // Size of HDR intensity.
    sizeof(float);
  auto devmem = alloc_mem(alloc_size);
  auto state_devmem = devmem.slice(0, denoiser_size.stateSizeInBytes);
  auto scratch_devmem = devmem.slice(denoiser_size.stateSizeInBytes,
    scratch_size);
  auto hdr_intensity_devmem = devmem.slice(hdr_intensity_offset, sizeof(float));
  return Denoiser {
    std::move(denoiser),
    std::move(devmem),
    std::move(state_devmem),
    std::move(scratch_devmem),
    std::move(hdr_intensity_devmem),
  };
  liong::log::info("destroyed denoiser");
}
void destroy_denoiser(Denoiser& denoiser) {
  free_mem(denoiser.devmem);
  OPTIX_ASSERT << optixDenoiserDestroy(denoiser.denoiser);
  denoiser = {};
  liong::log::info("destroyed denoiser");
}


void cmd_denoise(
  Transaction& transact,
  const Denoiser& denoiser,
  const Framebuffer& in_framebuf,
  const Framebuffer& out_framebuf,
  // Denoise the alpha channel or just copy the original alpha value?
  bool denoise_alpha
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
  params.denoiseAlpha = denoise_alpha;
  params.blendFactor = 0;
  params.hdrIntensity = denoiser.hdr_intensity_devmem.ptr;
  OptixImage2D in_img {};
  in_img.data = in_framebuf.framebuf_devmem.ptr;
  in_img.format = _fmt_lighar2optix(in_framebuf.fmt);
  in_img.width = in_framebuf.dim.x;
  in_img.height = in_framebuf.dim.y;
  // TODO: (penguinliong) See the programming guide for more information.
  //in_img.pixelStrideInBytes = in_framebuf.fmt.get_fmt_size();
  in_img.pixelStrideInBytes = 0;
  in_img.rowStrideInBytes = in_img.pixelStrideInBytes * in_img.width;
  OptixImage2D out_img {};
  out_img.data = out_framebuf.framebuf_devmem.ptr;
  out_img.format = _fmt_lighar2optix(out_framebuf.fmt);
  out_img.width = out_framebuf.dim.x;
  out_img.height = out_framebuf.dim.y;
  // out_img.pixelStrideInBytes = out_framebuf.fmt.get_fmt_size();
  out_img.pixelStrideInBytes = 0;
  out_img.rowStrideInBytes = out_img.pixelStrideInBytes * in_img.width;
  OPTIX_ASSERT << optixDenoiserInvoke(denoiser.denoiser, transact.stream,
    &params, denoiser.state_devmem.ptr, denoiser.state_devmem.size,
    &in_img, 1, 0, 0,
    &out_img, denoiser.scratch_devmem.ptr, denoiser.scratch_devmem.size);
  liong::log::info("scheduled framebuffer denoising");
}

} // namespace denoise

} // namespace liong
