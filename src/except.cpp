#include <sstream>
#include "except.hpp"

namespace liong {

CudaException::CudaException(CUresult code) {
  std::stringstream buf;
  const char* err_str;
  if (cuGetErrorString(code, &err_str) != CUDA_SUCCESS) {
    buf << "failed to describe error " << code;
  } else {
    buf << "cuda error: " << err_str;
  }
  msg = buf.str();
}
CudaException::CudaException(cudaError_t code) {
  std::stringstream buf;
  const char* err_str = cudaGetErrorString(code);
  buf << "cuda runtime error: " << err_str;
  msg = buf.str();
}
const char* CudaException::what() const {
  return msg.c_str();
}



OptixException::OptixException(OptixResult code) {
  std::stringstream buf;
  const char* err_str = optixGetErrorString(code);
  buf << "optix error: " << err_str;
  msg = buf.str();
}
const char* OptixException::what() const {
  return msg.c_str();
}



AssertionFailedException::AssertionFailedException(const std::string& msg) :
  msg(msg)  {}
const char* AssertionFailedException::what() const {
  return msg.c_str();
}


}
