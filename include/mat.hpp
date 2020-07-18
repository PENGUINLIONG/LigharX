#pragma once
// Material construction.
// @PENGUINLIONG
#include <vector>
#include <string>
#include "ty.hpp"

namespace liong {

enum MaterialEntryType {
  L_MATERIAL_TYPE_ALIGNMENT,
  L_MATERIAL_TYPE_TRIVIAL,
  L_MATERIAL_TYPE_BUFFER,
  L_MATERIAL_TYPE_TEXTURE_2D,
};


struct MaterialTrivialEntry {
  static const size_t MAX_TRIVIAL_SIZE = 16; // Size of a float4.

  size_t size;
  uint8_t data[MAX_TRIVIAL_SIZE];
};
struct MaterialBufferEntry {
  size_t size;
  const void* data;
};
struct MaterialTextureEntry {
  uint32_t w;
  uint32_t h;
  uint32_t d;
  const void* data;
  PixelFormat fmt;
};
struct MaterialAlignmentEntry {
  size_t align;
};

struct MaterialEntry {
  MaterialEntryType ty;
  size_t offset;
  union {
    MaterialTrivialEntry trivial;
    MaterialBufferEntry buf;
    MaterialTextureEntry tex;
    MaterialAlignmentEntry align;
  };
  MaterialEntry() {}
};

struct Material {
  DeviceMemory devmem;
};

// NOTE: Meterial injection order is the linear storage order.
struct MaterialBuilder {
  size_t max_align;
  size_t offset;
  size_t buffered_data_alloc_size;
  std::vector<MaterialEntry> entries;

  inline void _reserve_buf_alloc(size_t inc) {
    buffered_data_alloc_size = align_addr(buffered_data_alloc_size + inc,
      L_OPTIMAL_DEVMEM_ALIGN);
  }

  // Without adding more data, align the base address to specified number of
  // bytes. The value used for alignment will NOT be initialized.
  void push_align(size_t align) {
    max_align = std::max(align, max_align);
    MaterialEntry entry;
    entry.ty = L_MATERIAL_TYPE_ALIGNMENT;
    entry.offset = offset;
    entry.align.align = align;
    offset = align_addr(offset, align);
  }
  // Plain data that will be trivially copied to device memory. Trivial data
  // MUST NOT exceed `MaterialEntry::MAX_MATERIAL_ENTRY_SIZE`.
  template<typename T>
  void push_trivial(T var) {
    static_assert(sizeof(T) <= MaterialTrivialEntry::MAX_MATERIAL_ENTRY_SIZE,
      "type is too large as a material entry, try input by buffer instead");
    MaterialEntry entry;
    entry.ty = L_MATERIAL_TYPE_TRIVIAL;
    entry.offset = offset;
    entry.trivial.size = sizeof(T);
    *((T*)((const uint8_t*)entry.trivial.data)) = var;
    entries.emplace_back(std::move(entry));
    offset += sizeof(T);
  }
  // Device memory allocation, the input data buffer MUST be kept alive until
  // the material is built.
  void push_buf(const void* data, size_t size) {
    MaterialEntry entry;
    entry.ty = L_MATERIAL_TYPE_BUFFER;
    entry.offset = offset;
    entry.buf.size = size;
    entry.buf.data = data;
    entries.emplace_back(std::move(entry));
    offset += sizeof(CUdeviceptr);
    buffered_data_alloc_size += size;
  }
  // Sampled 2D texture, the input data buffer MUST be kept alive until the
  // material is built.
  void push_tex2d(
    const void* data,
    size_t size,
    uint32_t w,
    uint32_t h,
    PixelFormat fmt
  ) {
    MaterialEntry entry;
    entry.ty = L_MATERIAL_TYPE_TEXTURE_2D;
    entry.offset = offset;
    entry.tex.w = w;
    entry.tex.h = h;
    entry.tex.d = 1;
    entry.tex.data = data;
    entry.tex.fmt = fmt;
    entries.emplace_back(std::move(entry));
    offset += sizeof(CUtexObject);
    buffered_data_alloc_size += size;
  }

  Material build() {
    throw std::logic_error("todo");
  }

};

} // namespace liong
