#pragma once
// Argument parsing utilities.
// @PENGUINLIONG
#include <string>
#include "core.hpp"

namespace liong {

namespace args {

struct ArgumentParseConfig {
  uint32_t narg;
  bool (*parser)(const char*[], void*);
  void* dst;
};

// To use 

template<typename TTypedParser>
ArgumentParseConfig make_parse_cfg(void* dst) {
  ArgumentParseConfig parse_cfg;
  parse_cfg.narg = TTypedParser::narg;
  parse_cfg.dst = dst;
  parse_cfg.parser = &TTypedParser::parse;
  return parse_cfg;
}
void reg_arg(
  const char* short_flag,
  const char* long_flag,
  const ArgumentParseConfig& parse_cfg,
  const char* help
);
template<typename TTypedParser>
inline void reg_arg(
  const char* short_flag,
  const char* long_flag,
  typename TTypedParser::arg_ty& dst,
  const char* help
) {
  reg_arg(short_flag, long_flag, make_parse_cfg<TTypedParser>(&dst), help);
}


void parse_args(int argc, const char** argv);


//
// Parsers.
//


template<typename T>
struct TypedArgumentParser {
  typedef struct {} arg_ty;
  // Number of argument entries needed for this argument.
  static const uint32_t narg = -1;
  // Parser function. Convert the literal in the first parameter into structured
  // native representation. Return `true` on success.
  static bool parse(const char* lit[], void* dst) {
    return false;
  }
};
template<>
struct TypedArgumentParser<std::string> {
  typedef std::string arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(std::string*)dst = lit[0];
    return true;
  }
};
template<>
struct TypedArgumentParser<int> {
  typedef int arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(int*)dst = std::atoi(lit[0]);
    return true;
  }
};
template<>
struct TypedArgumentParser<float> {
  typedef float arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(float*)dst = std::atof(lit[0]);
    return true;
  }
};
template<>
struct TypedArgumentParser<int2> {
  typedef int arg_ty;
  static const uint32_t narg = 2;
  static bool parse(const char* lit[], void* dst) {
    *(int2*)dst = make_int2(std::atoi(lit[0]), std::atoi(lit[1]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float2> {
  typedef float arg_ty;
  static const uint32_t narg = 2;
  static bool parse(const char* lit[], void* dst) {
    *(float2*)dst = make_float2(std::atof(lit[0]), std::atof(lit[1]));
    return true;
  }
};
template<>
struct TypedArgumentParser<int3> {
  typedef int arg_ty;
  static const uint32_t narg = 3;
  static bool parse(const char* lit[], void* dst) {
    *(int3*)dst = make_int3(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float3> {
  typedef float arg_ty;
  static const uint32_t narg = 3;
  static bool parse(const char* lit[], void* dst) {
    *(float3*)dst = make_float3(std::atof(lit[0]), std::atof(lit[1]),
      std::atof(lit[2]));
    return true;
  }
};
template<>
struct TypedArgumentParser<int4> {
  typedef int arg_ty;
  static const uint32_t narg = 4;
  static bool parse(const char* lit[], void* dst) {
    *(int4*)dst = make_int4(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]), std::atoi(lit[3]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float4> {
  typedef float arg_ty;
  static const uint32_t narg = 4;
  static bool parse(const char* lit[], void* dst) {
    *(float4*)dst = make_float4(std::atof(lit[0]), std::atof(lit[1]),
      std::atof(lit[2]), std::atof(lit[3]));
    return true;
  }
};
// NOTE: This is used for arguments like `-f true` and `-f false`. If you need a
// boolean argument that don't need to be set explicitly. Use
// `SwitchArgumentParser` instead.
template<>
struct TypedArgumentParser<bool> {
  typedef bool arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    if (strcmp(lit[0], "true") == 0 || strcmp(lit[0], "True") == 0) {
      *(bool*)dst = true;
      return true;
    } else if (strcmp(lit[0], "false") == 0 || strcmp(lit[0], "False") == 0) {
      *(bool*)dst = false;
      return true;
    } else {
      return false;
    }
  }
};
struct SwitchArgumentParser {
  typedef bool arg_ty;
  static const uint32_t narg = 0;
  static bool parse(const char* lit[], void* dst) {
    *(bool*)dst = true;
    return true;
  }
};


using IntParser = TypedArgumentParser<int>;
using FloatParser = TypedArgumentParser<float>;
using BoolParser = TypedArgumentParser<bool>;
using StringParser = TypedArgumentParser<std::string>;
using SwitchParser = SwitchArgumentParser;

} // namespace args

} // namespace liong
