#include <algorithm>
#include "mat.hpp"

namespace liong {

MaterialBuilder& MaterialBuilder::align(size_t align) {
  max_align = std::max(align, max_align);
  size = align_addr(size, align);
  return *this;
}
MaterialBuilder& MaterialBuilder::with(const void* data, size_t size) {
  MaterialEntry entry;
  entry.offset = this->size;
  entry.size = size;
  std::memcpy(entry.data, data, size);
  entries.emplace_back(std::move(entry));
  this->size += size;
  return *this;
}
bool MaterialBuilder::build(void* data, size_t size) const {
  if (size < this->size) { return false; }
  for (const auto& entry : entries) {
    std::memcpy((uint8_t*)data + entry.offset, entry.data, entry.size);
  }
  return true;
}


} // namespace liong
