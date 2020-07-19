#include <algorithm>
#include "mat.hpp"

namespace liong {

MaterialBuilder& MaterialBuilder::align(size_t align) {
  max_align = std::max(align, max_align);
  offset = align_addr(offset, align);
  return *this;
}
MaterialBuilder& MaterialBuilder::with(const void* data, size_t size) {
  MaterialEntry entry;
  entry.offset = offset;
  entry.size = size;
  std::memcpy(entry.data, data, size);
  entries.emplace_back(std::move(entry));
  offset += size;
  return *this;
}
DeviceMemory MaterialBuilder::build() const {
  uint8_t* buf = (uint8_t*)std::malloc(offset);
  for (const auto& entry : entries) {
    std::memcpy(buf + entry.offset, entry.data, entry.size);
  }
  auto devmem = shadow_mem(buf, std::max(max_align, L_OPTIMAL_DEVMEM_ALIGN));
  std::free(buf);
  return std::move(devmem);
}


} // namespace liong
