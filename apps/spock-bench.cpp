#include "runtime/benchmark.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  spock::runtime::BenchmarkConfig config{};
  std::string output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) config.mode = spock::runtime::parse_bench_mode(argv[++i]);
    else if (arg == "--csv") config.json = false;
    else if (arg == "--json") config.json = true;
    else if (arg == "--warmup" && i + 1 < argc) config.warmup_runs = std::stoul(argv[++i]);
    else if (arg == "--runs" && i + 1 < argc) config.timed_runs = std::stoul(argv[++i]);
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--help") {
      std::cout << "usage: spock-bench --mode pp520|tg128|correctness [--json|--csv] [--warmup N] [--runs N] [--output PATH]\n";
      return 0;
    }
  }

  try {
    const auto result = spock::runtime::run_placeholder_benchmark(config);
    const auto rendered = config.json ? spock::runtime::render_json(result)
                                      : spock::runtime::render_csv(result);
    if (!output_path.empty()) {
      std::ofstream output(output_path);
      if (!output) {
        std::cerr << "spock-bench: unable to open output path: " << output_path << '\n';
        return 3;
      }
      output << rendered;
    } else {
      std::cout << rendered;
    }
  } catch (const std::exception& error) {
    std::cerr << "spock-bench: " << error.what() << '\n';
    return 2;
  }
}
