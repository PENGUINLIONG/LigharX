#pragma once
// Argument parsing utilities.
// @PENGUINLIONG
#include <string>
#include "core.hpp"

namespace liong {

namespace args {

struct ArgumentParseConfig {
  // Expected number of arguments segments.
  uint32_t narg;
  // Returns true if the parsing is successful.
  bool (*parser)(const char*[], void*);
  // Destination to be written with parsed value.
  void* dst;
};

// Optionally initialize argument parser with application name and usage
// description.
void init_arg_parse(const char* app_name, const char* desc);
// Print help message to the standard output.
void print_help();
// Erase the type of argument parser and bind the type-erased parser to the
// value destination. User code MUST ensure the `dst` buffer can contain the
// parsing result.
template<typename TTypedParser>
ArgumentParseConfig make_parse_cfg(void* dst) {
  ArgumentParseConfig parse_cfg;
  parse_cfg.narg = TTypedParser::narg;
  parse_cfg.dst = dst;
  parse_cfg.parser = &TTypedParser::parse;
  return parse_cfg;
}
// Register customized argument parsing.
void reg_arg(
  const char* short_flag,
  const char* long_flag,
  const ArgumentParseConfig& parse_cfg,
  const char* help
);
// Register a structural argument parsing.
template<typename TTypedParser>
inline void reg_arg(
  const char* short_flag,
  const char* long_flag,
  typename TTypedParser::arg_ty& dst,
  const char* help
) {
  reg_arg(short_flag, long_flag, make_parse_cfg<TTypedParser>(&dst), help);
}
// Parse arguments. Arguments will be matched against argument parsers
// registered before.
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
struct TypedArgumentParser<int32_t> {
  typedef int arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(int32_t*)dst = std::atoi(lit[0]);
    return true;
  }
};
template<>
struct TypedArgumentParser<uint32_t> {
  typedef int arg_ty;
  static const uint32_t narg = 1;
  static bool parse(const char* lit[], void* dst) {
    *(uint32_t*)dst = std::atoi(lit[0]);
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
  typedef int2 arg_ty;
  static const uint32_t narg = 2;
  static bool parse(const char* lit[], void* dst) {
    *(int2*)dst = make_int2(std::atoi(lit[0]), std::atoi(lit[1]));
    return true;
  }
};
template<>
struct TypedArgumentParser<uint2> {
  typedef uint2 arg_ty;
  static const uint32_t narg = 2;
  static bool parse(const char* lit[], void* dst) {
    *(uint2*)dst = make_uint2(std::atoi(lit[0]), std::atoi(lit[1]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float2> {
  typedef float2 arg_ty;
  static const uint32_t narg = 2;
  static bool parse(const char* lit[], void* dst) {
    *(float2*)dst = make_float2(std::atof(lit[0]), std::atof(lit[1]));
    return true;
  }
};
template<>
struct TypedArgumentParser<int3> {
  typedef int3 arg_ty;
  static const uint32_t narg = 3;
  static bool parse(const char* lit[], void* dst) {
    *(int3*)dst = make_int3(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]));
    return true;
  }
};
template<>
struct TypedArgumentParser<uint3> {
  typedef uint3 arg_ty;
  static const uint32_t narg = 3;
  static bool parse(const char* lit[], void* dst) {
    *(uint3*)dst = make_uint3(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float3> {
  typedef float3 arg_ty;
  static const uint32_t narg = 3;
  static bool parse(const char* lit[], void* dst) {
    *(float3*)dst = make_float3(std::atof(lit[0]), std::atof(lit[1]),
      std::atof(lit[2]));
    return true;
  }
};
template<>
struct TypedArgumentParser<int4> {
  typedef int4 arg_ty;
  static const uint32_t narg = 4;
  static bool parse(const char* lit[], void* dst) {
    *(int4*)dst = make_int4(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]), std::atoi(lit[3]));
    return true;
  }
};
template<>
struct TypedArgumentParser<uint4> {
  typedef uint4 arg_ty;
  static const uint32_t narg = 4;
  static bool parse(const char* lit[], void* dst) {
    *(uint4*)dst = make_uint4(std::atoi(lit[0]), std::atoi(lit[1]),
      std::atoi(lit[2]), std::atoi(lit[3]));
    return true;
  }
};
template<>
struct TypedArgumentParser<float4> {
  typedef float4 arg_ty;
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


using IntParser = TypedArgumentParser<int32_t>;
using Int2Parser = TypedArgumentParser<int2>;
using Int3Parser = TypedArgumentParser<int3>;
using Int4Parser = TypedArgumentParser<int4>;
using UintParser = TypedArgumentParser<uint32_t>;
using Uint2Parser = TypedArgumentParser<uint2>;
using Uint3Parser = TypedArgumentParser<uint3>;
using Uint4Parser = TypedArgumentParser<uint4>;
using FloatParser = TypedArgumentParser<float>;
using Float2Parser = TypedArgumentParser<float2>;
using Float3Parser = TypedArgumentParser<float3>;
using Float4Parser = TypedArgumentParser<float4>;
using BoolParser = TypedArgumentParser<bool>;
using StringParser = TypedArgumentParser<std::string>;
using SwitchParser = SwitchArgumentParser;

} // namespace args

} // namespace liong
