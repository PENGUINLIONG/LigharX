#include <map>
#include <vector>
#include <iostream>
#include "args.hpp"

namespace liong {

namespace args {

struct ArgumentHelp {
  std::string short_flag;
  std::string long_flag;
  std::string help;
};
struct ArgumentConfig {
  std::string app_name = "[APPNAME]";
  std::string desc;
  // Short flag name -> ID.
  std::map<char, size_t> short_map;
  // Long flag name -> ID.
  std::map<std::string, size_t> long_map;
  // Argument parsing info.
  std::vector<ArgumentParseConfig> parse_cfgs;
  // Argument help info.
  std::vector<ArgumentHelp> helps;
} arg_cfg;



void init_arg_parse(const char* app_name, const char* desc) {
  arg_cfg.app_name = app_name;
  arg_cfg.desc = desc;
}
void print_help() {
  std::cout << "usage: " << arg_cfg.app_name << " [OPTIONS]" << std::endl;
  for (const auto& help : arg_cfg.helps) {
    std::cout << help.short_flag << "\t"
      << help.long_flag << "\t\t"
      << help.help << std::endl;
  }
  std::cout << "-h\t--help\t\tPrint this message." << std::endl;
  if (!arg_cfg.desc.empty()) {
    std::cout << arg_cfg.desc << std::endl;
  }
  std::exit(0);
}
void report_unknown_arg(const char* arg) {
  std::cout << arg << std::endl;
  print_help();
}

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
      ASSERT << parse_cfg.parser(argv + i, parse_cfg.dst)
        << "unable to parse argument";
      ASSERT << (argc - i >= parse_cfg.narg)
        << "no enough argument segments";
      i += parse_cfg.narg;
      iarg_entry = -1;
    } else {
      const char* arg = argv[i];
      if (arg[0] != '-') {
        // Free argument.
        ASSERT << false
          << "free argument is currently unsupported";
      } else if (arg[1] != '-') {
        if (arg[1] == 'h') { print_help(); }
        // Short flag argument.
        auto it = arg_cfg.short_map.find(arg[1]);
        if (it != arg_cfg.short_map.end()) {
          iarg_entry = it->second;
        } else {
          report_unknown_arg(arg);
        }
        ++i;
      } else {
        if (std::strcmp(arg + 2, "help") == 0) { print_help(); }
        // Long flag argument.
        auto it = (arg_cfg.long_map.find(arg + 2));
        if (it != arg_cfg.long_map.end()) {
          iarg_entry = it->second;
        } else {
          report_unknown_arg(arg);
        }
        ++i;
      }
    }
  }
}


} // namespace args

} // namespace liong


