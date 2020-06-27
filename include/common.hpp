#pragma once
// Common utilities.
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>


namespace liong {

namespace log {
// Logging infrastructure.

enum class LogLevel {
  INFO,
  WARNING,
  ERROR,
};

namespace detail {

extern void (*log_callback)(LogLevel lv, const std::string& msg);

template<typename T, typename ... TArgs>
inline void gen_log_impl(
  std::stringstream& ss,
  const T& first,
  const TArgs& ... msg
) {
  gen_log_impl(ss, first);
  gen_log_impl(ss, msg...);
}
template<typename T>
inline void gen_log_impl(std::stringstream& ss, const T& msg) {
  ss << msg;
}

}


void set_log_callback(decltype(detail::log_callback) cb);
template<typename ... TArgs>
void log(LogLevel level, const TArgs& ... msg) {
  std::stringstream ss {};
  detail::gen_log_impl(ss, msg...);
  detail::log_callback(level, ss.str());
}

template<typename ... TArgs>
inline void info(const TArgs& ... msg) {
  log(LogLevel::INFO, msg...);
}
template<typename ... TArgs>
inline void warn(const TArgs& ... msg) {
  log(LogLevel::WARNING, msg...);
}
template<typename ... TArgs>
inline void error(const TArgs& ... msg) {
  log(LogLevel::ERROR, msg...);
}
}


}
