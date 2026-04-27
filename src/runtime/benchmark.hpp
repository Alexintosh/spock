#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace spock::runtime {

enum class BenchMode {
  Pp520,
  Tg128,
  ShortCorrectness,
};

struct BenchmarkConfig {
  BenchMode mode = BenchMode::Tg128;
  std::uint32_t warmup_runs = 3;
  std::uint32_t timed_runs = 10;
  bool json = true;
};

struct BenchmarkResult {
  BenchMode mode;
  std::vector<double> run_ms;
  double mean_ms = 0.0;
  double tokens_per_second = 0.0;
};

BenchMode parse_bench_mode(std::string_view value);
std::string_view to_string(BenchMode mode);
BenchmarkResult run_placeholder_benchmark(const BenchmarkConfig& config);
std::string render_json(const BenchmarkResult& result);
std::string render_csv(const BenchmarkResult& result);

}  // namespace spock::runtime
