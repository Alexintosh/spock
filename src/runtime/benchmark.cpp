#include "runtime/benchmark.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace spock::runtime {

BenchMode parse_bench_mode(const std::string_view value) {
  if (value == "pp520") return BenchMode::Pp520;
  if (value == "tg128") return BenchMode::Tg128;
  if (value == "correctness") return BenchMode::ShortCorrectness;
  throw std::invalid_argument("unknown benchmark mode: " + std::string(value));
}

std::string_view to_string(const BenchMode mode) {
  switch (mode) {
    case BenchMode::Pp520:
      return "pp520";
    case BenchMode::Tg128:
      return "tg128";
    case BenchMode::ShortCorrectness:
      return "correctness";
  }
  return "unknown";
}

static std::uint32_t token_count(const BenchMode mode) {
  switch (mode) {
    case BenchMode::Pp520:
      return 520;
    case BenchMode::Tg128:
      return 128;
    case BenchMode::ShortCorrectness:
      return 16;
  }
  return 0;
}

BenchmarkResult run_placeholder_benchmark(const BenchmarkConfig& config) {
  for (std::uint32_t i = 0; i < config.warmup_runs; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  BenchmarkResult result{.mode = config.mode};
  result.run_ms.reserve(config.timed_runs);
  for (std::uint32_t i = 0; i < config.timed_runs; ++i) {
    const auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto end = std::chrono::steady_clock::now();
    result.run_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  result.mean_ms = std::accumulate(result.run_ms.begin(), result.run_ms.end(), 0.0) /
                   static_cast<double>(result.run_ms.size());
  result.tokens_per_second = 1000.0 * static_cast<double>(token_count(config.mode)) / result.mean_ms;
  return result;
}

std::string render_json(const BenchmarkResult& result) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"mode\": \"" << to_string(result.mode) << "\",\n";
  out << "  \"mean_ms\": " << result.mean_ms << ",\n";
  out << "  \"tokens_per_second\": " << result.tokens_per_second << ",\n";
  out << "  \"implementation\": \"placeholder-cpu-timer\"\n";
  out << "}\n";
  return out.str();
}

std::string render_csv(const BenchmarkResult& result) {
  std::ostringstream out;
  out << "mode,mean_ms,tokens_per_second,implementation\n";
  out << to_string(result.mode) << ',' << result.mean_ms << ',' << result.tokens_per_second
      << ",placeholder-cpu-timer\n";
  return out.str();
}

}  // namespace spock::runtime
