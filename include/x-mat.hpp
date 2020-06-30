#pragma once
// CrossMaterial 
// @PENGUINLIONG
#include <x.hpp>

namespace liong {

#ifdef __CUDACC__
#define GET_MATERIAL_PTR(ty) (*(const ty**)                                    \
  ((const char*)optixGetSbtDataPointer() + OPTIX_SBT_RECORD_HEADER_SIZE))
#endif

struct HitMaterial {
  vec3 obj_color;
};
struct MissMaterial {
  vec3 sky_color;
};

}
