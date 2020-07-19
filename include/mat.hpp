#pragma once
// Material construction utilities.
// @PENGUINLIONG
#include <functional>
#include <vector>
#include "ty.hpp"
#include "core.hpp"

namespace liong {

// NOTE: Meterial injection order is the linear storage order.
class MaterialBuilder {
private:
  struct MaterialEntry {
    static const size_t MAX_MATERIAL_ENTRY_SIZE = 16; // Size of a float4.

    size_t offset;
    size_t size;
    uint8_t data[MAX_MATERIAL_ENTRY_SIZE];
  };
  std::vector<MaterialEntry> entries;
  MaterialBuilder& with(const void* data, size_t size);

public:
  // You SHOULD use this field.
  size_t max_align = L_OPTIMAL_DEVMEM_ALIGN;
  // You SHOULD use this field.
  size_t size;

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
  MaterialBuilder& align(size_t align);
  // Push a trivial value of any type within the valid size as specified
  // previously, at the end of all existing material entries. It's worth
  // noticing that this method also accept device memory pointers and texture
  // object handles.
  template<typename T>
  inline MaterialBuilder& with(T var) {
    static_assert(sizeof(T) <= MaterialEntry::MAX_MATERIAL_ENTRY_SIZE,
      "type is too large as a material entry, try input by buffer instead");
    return with(&var, sizeof(var));
  }

  // Create a host memory contains material data. Returns true if the buffer can
  // contain all the material data.
  //
  // WARNING: The preset alignment will only work if you shadow the built
  // material buffer with alignemnt of `max_align`.
  bool build(void* data, size_t size) const;
};

inline DeviceMemory build_mat(
  const std::function<void(MaterialBuilder&)>& mat_build_fn
) {
  MaterialBuilder mat_build {};
  std::vector<uint8_t> mat_buf {};
  mat_build_fn(mat_build);
  mat_buf.resize(mat_build.size);
  mat_build.build((void*)mat_buf.data(), mat_buf.size());
  return shadow_mem(mat_buf.data(), mat_buf.size(), mat_build.max_align);
}

} // namespace liong
