#include <algorithm>
#include "mat.hpp"

namespace liong {

namespace mat {

void align_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t align
) {
  constexpr size_t max_acceptable_align =
    std::min(OPTIX_SBT_RECORD_ALIGNMENT, OPTIX_SBT_RECORD_HEADER_SIZE);
  ASSERT << (align <= max_acceptable_align)
    << "material entry alignment is too large";
  mat_ty.max_align = std::max(align, mat_ty.max_align);
  mat_ty.size = align_addr(mat_ty.size, align);
}
void push_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t size
) {
  ASSERT << (size <= MaterialEntry::MAX_MATERIAL_ENTRY_SIZE)
    << "type is too large as a material entry, try input by buffer instead";
  mat_ty.entries[name] = MaterialEntry { mat_ty.size, size };
  mat_ty.size += size;
}
void add_mat_ty_entry(
  MaterialType& mat_ty,
  const char* name,
  size_t offset,
  size_t size
) {
  ASSERT << (size <= MaterialEntry::MAX_MATERIAL_ENTRY_SIZE)
    << "type is too large as a material entry, try input by buffer instead";
  mat_ty.entries[name] = MaterialEntry { offset, size };
  mat_ty.size = std::max(offset, mat_ty.size) + size;
}



Material create_mat(MaterialType& mat_ty) {
  return Material { std::malloc(mat_ty.size), mat_ty.size, mat_ty.max_align };
}
void destroy_mat(Material& mat) {
  if (mat.data) { std::free(mat.data); }
  mat = {};
}
void assign_mat_entry(
  const MaterialType& mat_ty,
  const Material& mat,
  const char* name,
  const void* data,
  size_t size
) {
  const auto& entry = mat_ty.entries.at(name);
  ASSERT << (size == entry.size)
    << "cannot assign material variable with mismatched size";
  std::memcpy(((uint8_t*)mat.data) + entry.offset, data, size);
}

} // namespace mat

} // namespace liong
