#pragma once
// Result checks and runtime assertion.
// @PENGUINLIONG
#include <exception>
#include <string>
#include "x.hpp"

namespace liong {

class CudaException : public std::exception {
  std::string msg;
public:
  CudaException(CUresult code);
  CudaException(cudaError_t code);

  const char* what() const override;
};
struct CudaAssert {
  inline const CudaAssert& operator<<(CUresult code) const {
    if (code != CUDA_SUCCESS) { throw CudaException(code); }
    return *this;
  }
  inline const CudaAssert& operator<<(cudaError_t code) const {
    if (code != CUDA_SUCCESS) { throw CudaException(code); }
    return *this;
  }
};
#define CUDA_ASSERT (::liong::CudaAssert{})



class OptixException : public std::exception {
  std::string msg;
public:
  OptixException(OptixResult code);

  const char* what() const override;
};
struct OptixAssert {
  inline const OptixAssert& operator<<(OptixResult code) const {
    if (code != OPTIX_SUCCESS) { throw OptixException(code); }
    return *this;
  }
};
#define OPTIX_ASSERT (::liong::OptixAssert{})



class AssertionFailedException : public std::exception {
  std::string msg;
public:
  AssertionFailedException(const std::string& msg);

  const char* what() const override;
};
struct Asserted { bool cond; };
struct Assert {};
inline const Assert operator<<(Asserted a, const std::string& msg) {
  if (a.cond) {
    return {};
  } else {
    throw AssertionFailedException { msg };
  }
}
inline const Asserted operator<<(Assert a, bool cond) {
  return Asserted { cond };
}
#define ASSERT (::liong::Assert{})

}
