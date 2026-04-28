#include "runtime/vk_decode.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  spock::runtime::DecodeConfig config;
  bool use_stdin = false;
  std::string token_file;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--repack-dir" && i + 1 < argc) {
      config.repack_dir = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      config.prompt_text = argv[++i];
    } else if (arg == "--tokens" && i + 1 < argc) {
      // Read token IDs from a file (space-separated)
      token_file = argv[++i];
    } else if (arg == "--max-new-tokens" && i + 1 < argc) {
      config.max_new_tokens = std::stoul(argv[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      config.verbose = true;
    } else if (arg == "--stdin") {
      use_stdin = true;
    } else if (arg == "--help") {
      std::cout << "usage: spock-decode --repack-dir DIR [options]\n";
      std::cout << "  --prompt TEXT       Prompt text\n";
      std::cout << "  --tokens FILE       File with space-separated token IDs\n";
      std::cout << "  --stdin             Read prompt from stdin\n";
      std::cout << "  --max-new-tokens N  Tokens to generate (default 16)\n";
      std::cout << "  --verbose / -v      Verbose output\n";
      return 0;
    }
  }

  if (config.repack_dir.empty()) {
    std::cerr << "spock-decode: --repack-dir is required\n";
    return 1;
  }

  // Load token IDs if provided
  if (!token_file.empty()) {
    std::ifstream f(token_file);
    if (!f) {
      std::cerr << "spock-decode: cannot open token file: " << token_file << "\n";
      return 2;
    }
    uint32_t tok;
    while (f >> tok) {
      config.prompt_tokens.push_back(tok);
    }
  }

  if (use_stdin) {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty()) {
        config.prompt_text = line;
        break;
      }
    }
  }

  if (config.prompt_tokens.empty() && config.prompt_text.empty()) {
    // Default: use a simple prompt token sequence
    config.prompt_tokens = {9419, 11, 1814, 0};  // "Hello, world!"
  }

  std::cout << "{\n";
  std::cout << "  \"repack_dir\": \"" << config.repack_dir << "\",\n";
  std::cout << "  \"max_new_tokens\": " << config.max_new_tokens << ",\n";

  auto result = spock::runtime::run_vk_decode(config);

  if (!result.error.empty()) {
    std::cout << "  \"error\": \"" << result.error << "\"\n";
    std::cout << "}\n";
    return 3;
  }

  std::cout << "  \"prompt_tokens\": [";
  for (size_t i = 0; i < result.prompt_tokens.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << result.prompt_tokens[i];
  }
  std::cout << "],\n";

  std::cout << "  \"generated_tokens\": [";
  for (size_t i = 0; i < result.generated_tokens.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << result.generated_tokens[i];
  }
  std::cout << "],\n";

  std::cout << "  \"generated_count\": " << result.generated_tokens.size() << ",\n";
  std::cout << "  \"elapsed_ms\": " << result.elapsed_ms << ",\n";
  std::cout << "  \"status\": \"ok\"\n";
  std::cout << "}\n";

  return 0;
}
