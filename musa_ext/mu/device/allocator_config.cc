/* Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mu/device/allocator_config.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace tensorflow {
namespace musa {

namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string Trim(const std::string& s) {
  const auto is_space = [](unsigned char c) { return std::isspace(c); };
  auto b = std::find_if_not(s.begin(), s.end(), is_space);
  auto e = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
  return (b >= e) ? std::string() : std::string(b, e);
}

bool ParseBool(const std::string& v) {
  const std::string s = ToLower(v);
  return s == "1" || s == "true" || s == "t" || s == "yes" || s == "y" ||
         s == "on";
}

// Parses an integer with optional unit: raw bytes, or an integer
// suffixed by `kb`/`mb`/`gb` (case-insensitive). Returns 0 on parse
// failure, which downstream treats as "unset / unlimited".
std::uint64_t ParseByteSize(const std::string& v) {
  if (v.empty()) return 0;
  char* end = nullptr;
  long long base = std::strtoll(v.c_str(), &end, 10);
  if (base < 0) return 0;
  std::string suffix = ToLower(std::string(end != nullptr ? end : ""));
  suffix = Trim(suffix);
  if (suffix.empty() || suffix == "b") {
    return static_cast<std::uint64_t>(base);
  }
  if (suffix == "kb" || suffix == "k")
    return static_cast<std::uint64_t>(base) * 1024ull;
  if (suffix == "mb" || suffix == "m") {
    return static_cast<std::uint64_t>(base) * 1024ull * 1024ull;
  }
  if (suffix == "gb" || suffix == "g") {
    return static_cast<std::uint64_t>(base) * 1024ull * 1024ull * 1024ull;
  }
  return 0;
}

}  // namespace

const AllocatorConfig& AllocatorConfig::Instance() {
  // Leaked to match the other singletons in libmusa_core — we need
  // access during process teardown paths.
  static AllocatorConfig* inst = []() {
    auto* c = new AllocatorConfig();
    c->ParseFromEnv();
    return c;
  }();
  return *inst;
}

void AllocatorConfig::ParseFromEnv() {
  const char* env = std::getenv("TF_MUSA_ALLOC_CONF");
  if (env == nullptr || *env == '\0') return;
  raw_ = env;

  // Split comma-delimited `key:value` pairs. Whitespace around keys
  // and values is tolerated to match torch's parser.
  std::string s(raw_);
  std::size_t i = 0;
  while (i <= s.size()) {
    std::size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    std::string pair = Trim(s.substr(i, j - i));
    if (!pair.empty()) {
      std::size_t colon = pair.find(':');
      // Accept `=` too since some docs use it; be permissive.
      if (colon == std::string::npos) colon = pair.find('=');
      if (colon == std::string::npos) {
        std::fprintf(stderr,
                     "[MUSA] TF_MUSA_ALLOC_CONF: ignoring malformed entry "
                     "'%s' (expected key:value)\n",
                     pair.c_str());
      } else {
        std::string k = ToLower(Trim(pair.substr(0, colon)));
        std::string v = Trim(pair.substr(colon + 1));

        if (k == "expandable_segments") {
          expandable_segments_ = ParseBool(v);
        } else if (k == "max_split_size_mb") {
          // Canonical PyTorch spelling is "mb"; multiply accordingly.
          char* end = nullptr;
          long long n = std::strtoll(v.c_str(), &end, 10);
          if (n > 0) {
            max_split_size_bytes_ =
                static_cast<std::uint64_t>(n) * 1024ull * 1024ull;
          }
        } else if (k == "max_split_size_bytes") {
          max_split_size_bytes_ = ParseByteSize(v);
        } else if (k == "roundup_power2_divisions") {
          roundup_power2_divisions_ = std::atoi(v.c_str());
        } else if (k == "garbage_collection_threshold") {
          garbage_collection_threshold_ = std::atof(v.c_str());
        } else {
          std::fprintf(stderr,
                       "[MUSA] TF_MUSA_ALLOC_CONF: ignoring unknown key "
                       "'%s'\n",
                       k.c_str());
        }
      }
    }
    if (j == s.size()) break;
    i = j + 1;
  }
}

}  // namespace musa
}  // namespace tensorflow
