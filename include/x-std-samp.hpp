// # Sampling Utilities
// @penguinliong
//
// Standard even-spacing sampling utilities. The following sampling points are
// the referential sampling patterns defined by the Vulkan Specification.
// See Section 24.3. Multisampling of the Vulkan Specification for more
// information.
#include <x.hpp>

namespace std_samp1 {

constexpr const size_t NSAMP_PT = 1;
CONST_BUF float2 samp_pts[1] = {
  { 0.5, 0.5 },
};

}
namespace std_samp2 {

constexpr const size_t NSAMP_PT = 2;
CONST_BUF float2 samp_pts[2] = {
  { 0.75, 0.75 },
  { 0.25, 0.25 },
};

}
namespace std_samp4 {

static constexpr const size_t NSAMP_PT = 4;
CONST_BUF float2 samp_pts[4] = {
  { 0.375, 0.125 },
  { 0.875, 0.375 },
  { 0.125, 0.625 },
  { 0.625, 0.875 },
};

}
namespace std_samp8 {

constexpr const size_t NSAMP_PT = 8;
CONST_BUF float2 samp_pts[8] = {
  { 0.5625, 0.3125 },
  { 0.4375, 0.6875 },
  { 0.8125, 0.5625 },
  { 0.3125, 0.1875 },
  { 0.1875, 0.8125 },
  { 0.0625, 0.4375 },
  { 0.6875, 0.9375 },
  { 0.9375, 0.0625 },
};

}
namespace std_samp16 {

constexpr const size_t NSAMP_PT = 16;
CONST_BUF float2 samp_pts[16] = {
  { 0.5625, 0.5625 },
  { 0.4375, 0.3125 },
  { 0.3125, 0.625 },
  { 0.75, 0.4375 },
  { 0.1875, 0.375 },
  { 0.625, 0.8125 },
  { 0.8125, 0.6875 },
  { 0.6875, 0.1875 },
  { 0.375, 0.875 },
  { 0.5, 0.0625 },
  { 0.25, 0.125 },
  { 0.125, 0.75 },
  { 0.0, 0.5 },
  { 0.9375, 0.25 },
  { 0.875, 0.9375 },
  { 0.0625, 0.0 },
};

}
