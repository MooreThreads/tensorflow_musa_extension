#pragma once

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>

// #if defined(__CUDACC__) || defined(__MUSA__)
#define MUSA_HOST_DEVICE __host__ __device__
// #else
// #define MUSA_HOST_DEVICE
// #endif

namespace tensorflow {
namespace random {

class PhiloxRandom {
 public:
  static constexpr int kResultElementCount = 4;

  MUSA_HOST_DEVICE PhiloxRandom(uint64_t key0 = 0, uint64_t key1 = 0,
                                uint64_t counter = 0)
      : key0_(key0), key1_(key1), counter_(counter) {}

  MUSA_HOST_DEVICE void Skip(uint64_t skip) { counter_ += skip; }

  MUSA_HOST_DEVICE void Next(uint32_t result[kResultElementCount]) {
    FillBlock(counter_, result);
    ++counter_;
  }

 private:
  MUSA_HOST_DEVICE static uint64_t SplitMix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  MUSA_HOST_DEVICE void FillBlock(uint64_t index,
                                  uint32_t result[kResultElementCount]) const {
    uint64_t state = index ^ key0_;
    uint64_t mix = SplitMix64(state + key1_);
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = static_cast<uint32_t>(mix);
      mix = SplitMix64(mix + static_cast<uint64_t>(i) + key1_);
    }
  }

  uint64_t key0_;
  uint64_t key1_;
  uint64_t counter_;
};

template <typename Generator>
class NormalDistribution {
 public:
  static constexpr int kResultElementCount = Generator::kResultElementCount;

  MUSA_HOST_DEVICE NormalDistribution(double mean = 0.0, double stddev = 1.0)
      : mean_(mean), stddev_(stddev) {
    static_assert(kResultElementCount % 2 == 0,
                  "NormalDistribution requires an even result count");
  }

  struct Result {
    double values[kResultElementCount];
    MUSA_HOST_DEVICE double& operator[](int index) { return values[index]; }
    MUSA_HOST_DEVICE const double& operator[](int index) const {
      return values[index];
    }
  };

  MUSA_HOST_DEVICE Result operator()(Generator* generator) const {
    Result result;
    uint32_t raw[kResultElementCount];
    generator->Next(raw);
    FillResult(result, raw);
    return result;
  }

 private:
  static constexpr double kTwoPi = 6.283185307179586476925286766559;
  static constexpr double kMinUniform = 1e-30;

  MUSA_HOST_DEVICE static double Uniform(uint32_t value) {
    constexpr double kInv =
        1.0 / (static_cast<double>(std::numeric_limits<uint32_t>::max()) + 1.0);
    return std::max(kMinUniform,
                    (static_cast<double>(value) + 0.5) * kInv);
  }

  MUSA_HOST_DEVICE void FillResult(Result& result,
                                   const uint32_t raw[kResultElementCount]) const {
    const double base = static_cast<double>(mean_);
    const double scale = static_cast<double>(stddev_);
    for (int pair = 0; pair < kResultElementCount / 2; ++pair) {
      const double u = Uniform(raw[pair * 2]);
      const double v = Uniform(raw[pair * 2 + 1]);
      const double radius = std::sqrt(-2.0 * std::log(u));
      const double angle = kTwoPi * v;

      result[pair * 2] = base + scale * (radius * std::cos(angle));
      result[pair * 2 + 1] = base + scale * (radius * std::sin(angle));
    }
  }

  double mean_;
  double stddev_;
};

template <typename Generator>
class TruncatedNormalDistribution : public NormalDistribution<Generator> {
 public:
  using Base = NormalDistribution<Generator>;
  using Result = typename Base::Result;
  using Base::kResultElementCount;

  MUSA_HOST_DEVICE TruncatedNormalDistribution(double mean = 0.0, double stddev = 1.0,
                                              double truncation = 2.0)
      : Base(mean, stddev),
        center_(mean),
        limit_(std::fabs(truncation) * stddev) {}

  MUSA_HOST_DEVICE Result operator()(Generator* generator) const {
    Result result;
    int filled = 0;
    constexpr int kMaxIterations = 100;  // Prevent infinite loop
    int iterations = 0;

    while (filled < kResultElementCount && iterations < kMaxIterations) {
      auto candidate = Base::operator()(generator);
      for (int i = 0; i < kResultElementCount && filled < kResultElementCount;
           ++i) {
        const double value = candidate[i];
        if (std::fabs(value - center_) <= limit_) {
          result[filled++] = value;
        }
      }
      ++iterations;
    }

    // Fill remaining with boundary values if max iterations reached
    while (filled < kResultElementCount) {
      result[filled++] = center_;
    }

    return result;
  }

 private:
  double center_;
  double limit_;  // Already scaled by stddev
};

}  // namespace random

class GuardedPhiloxRandom {
 public:
  GuardedPhiloxRandom() = default;

  void Init(int64_t seed, int64_t seed2) {
    std::lock_guard<std::mutex> guard(mu_);
    key0_ = static_cast<uint64_t>(seed);
    key1_ = static_cast<uint64_t>(seed2);
    next_group_ = 0;
    initialized_ = true;
  }

  random::PhiloxRandom ReserveSamples32(int64_t samples) {
    const uint64_t safe_samples =
        samples < 0 ? 0 : static_cast<uint64_t>(samples);
    const uint64_t groups = (safe_samples +
                             random::PhiloxRandom::kResultElementCount - 1) /
                            random::PhiloxRandom::kResultElementCount;
    std::lock_guard<std::mutex> guard(mu_);
    if (!initialized_) {
      key0_ = 0;
      key1_ = 0;
      next_group_ = 0;
      initialized_ = true;
    }
    uint64_t base = next_group_;
    next_group_ += groups;
    return random::PhiloxRandom(key0_, key1_, base);
  }

 private:
  mutable std::mutex mu_;
  uint64_t key0_ = 0;
  uint64_t key1_ = 0;
  uint64_t next_group_ = 0;
  bool initialized_ = false;
};

}  // namespace tensorflow
