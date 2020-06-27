#include "common.hpp"

namespace liong {

namespace log {
namespace detail {
decltype(log_callback) log_callback = nullptr;
}
void set_log_callback(decltype(detail::log_callback) cb) {
  detail::log_callback = cb;
}
}


}
