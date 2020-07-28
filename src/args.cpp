#include <map>
#include <vector>
#include "args.hpp"

namespace liong {

namespace args {

struct ArgumentHelp {
  std::string short_flag;
  std::string long_flag;
  std::string help;
};
struct ArgumentConfig {
  // Short flag name -> ID.
  std::map<char, size_t> short_map;
  // Long flag name -> ID.
  std::map<std::string, size_t> long_map;
  // Argument parsing info.
  std::vector<ArgumentParseConfig> parse_cfgs;
  // Argument help info.
  std::vector<ArgumentHelp> helps;
} arg_cfg;



void reg_arg(
  const char* short_flag,
  const char* long_flag,
  const ArgumentParseConfig& parse_cfg,
  const char* help
) {
  using std::strlen;
  size_t i = arg_cfg.parse_cfgs.size();
  if (strlen(short_flag) == 2 && short_flag[0] == '-') {
    arg_cfg.short_map[short_flag[1]] = i;
  }
  if (strlen(long_flag) > 3 && long_flag[1] == '-' && long_flag[0] == '-') {
    arg_cfg.long_map[long_flag + 2] = i;
  }
  arg_cfg.parse_cfgs.emplace_back(parse_cfg);
  ArgumentHelp arg_help { short_flag, long_flag, help };
  arg_cfg.helps.emplace_back(std::move(arg_help));
}



void parse_args(int argc, const char** argv) {
  auto i = 1;
  int iarg_entry = -1;
  while (i < argc) {
    if (iarg_entry >= 0) {
      auto& parse_cfg = arg_cfg.parse_cfgs[iarg_entry];
      auto j = parse_cfg.narg;
      ASSERT << !parse_cfg.parser(argv + i, parse_cfg.dst)
        << "unable to parse argument";
      i += parse_cfg.narg;
      iarg_entry = -1;
    } else {
      const char* arg = argv[i];
      if (arg[0] != '-') {
        // Free argument.
        ASSERT << false
          << "free argument is currently unsupported";
      } else if (arg[1] != '-') {
        // Short flag argument.
        iarg_entry = arg_cfg.short_map[arg[1]];
        ++i;
      } else {
        // Long flag argument.
        iarg_entry = arg_cfg.long_map[arg + 2];
        ++i;
      }
    }
  }
}



} // namespace args

} // namespace liong


