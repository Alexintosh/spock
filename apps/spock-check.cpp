#include "model/qwen35_config.hpp"
#include "reference/qwen35_cpu_reference.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string prompt_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "usage: spock-check\n";
      std::cout << "       spock-check --prompts tests/data/prompts.jsonl\n";
      std::cout << "Runs deterministic model-contract and CPU-reference smoke checks.\n";
      return 0;
    }
    if (arg == "--prompts" && i + 1 < argc) {
      prompt_path = argv[++i];
    }
  }

  const auto& schedule = spock::model::Qwen35Config::layer_schedule();
  std::size_t deltanet = 0;
  std::size_t attention = 0;
  for (const auto kind : schedule) {
    if (kind == spock::model::LayerKind::DeltaNet) ++deltanet;
    if (kind == spock::model::LayerKind::FullAttention) ++attention;
  }

  spock::reference::Qwen35CpuReference ref;
  const std::uint32_t prompt[] = {1, 2, 3};
  const auto tokens = ref.decode(prompt, {.max_new_tokens = 4});

  std::size_t prompt_count = 0;
  if (!prompt_path.empty()) {
    std::ifstream input(prompt_path);
    if (!input) {
      std::cerr << "spock-check: unable to open prompt corpus: " << prompt_path << '\n';
      return 2;
    }
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty()) ++prompt_count;
    }
    if (prompt_count < 32) {
      std::cerr << "spock-check: prompt corpus must contain at least 32 prompts\n";
      return 3;
    }
  }

  std::cout << "{\n";
  std::cout << "  \"model\": \"" << spock::model::Qwen35Config::model_id << "\",\n";
  std::cout << "  \"layers\": " << schedule.size() << ",\n";
  std::cout << "  \"deltanet_layers\": " << deltanet << ",\n";
  std::cout << "  \"attention_layers\": " << attention << ",\n";
  std::cout << "  \"prompt_count\": " << prompt_count << ",\n";
  std::cout << "  \"reference_tokens\": " << tokens.size() << ",\n";
  std::cout << "  \"status\": \"ok\"\n";
  std::cout << "}\n";
}
