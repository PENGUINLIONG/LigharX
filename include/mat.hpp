#pragma once
// Material construction utilities.
// @PENGUINLIONG
#include <string>
#include <map>
#include "ty.hpp"
#include "core.hpp"

namespace liong {

namespace mat {

struct MaterialEntry {
  static const size_t MAX_MATERIAL_ENTRY_SIZE = 16; // Size of a float4.

  size_t offset;
  size_t size;
};
// Material variable manager.
struct MaterialType {
  std::map<std::string, MaterialEntry> entries;
  size_t size;
  size_t max_align;
};

// This will not add any new data but the global address of the next pushed
// material entry will start from an address aligned to `align`. The memory
// padding has undefined content and SHOULD NOT be accessed anyway. The final
// allocated device memory will be aligned to the maximum alignment the entire
// object. The material buffer will at least be aligned to
// `L_OPTIMAL_DEVMEM_ALIGN`.
//
// WARNING: This method doesn't check but all the required alignment MUST be
// in the same power series, typically and most efficiently they are power of
// two.
extern void align_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t align
);
// Reserve a space for trivial value of any type within the valid size as
// specified previously, at the end of all existing material entries. It's
// worth noticing that this method also accept device memory pointers and
// texture object handles.
extern void push_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t size
);
// Directly specify the offset and size of desired material variable
// positioning. This will allow user code to create `union`s.
extern void add_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t offset,
  size_t size
);



// Material data buffer on the host side.
struct Material {
  void* data;
  size_t size;
  size_t align;
};

// Create a host memory contains material data. Returns true if the buffer can
// contain all the material data.
//
// WARNING: The preset alignment will only work if you shadow the built
// material buffer to the device side with alignemnt of `align`.
extern Material create_mat(MaterialType& mat_ty);
extern void destroy_mat(Material& mat);
// Assign an material variable according to the placement specified by `mat_ty`.
extern void assign_mat_entry(
  const MaterialType& mat_ty,
  const Material& mat,
  const char* name,
  const void* data,
  size_t size
);

} // namespace mat

} // namespace liong
