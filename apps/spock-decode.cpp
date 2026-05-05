#include "runtime/vk_decode.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
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
    } else if (arg == "--debug-dump") {
      config.debug_dump = true;
    } else if (arg == "--stdin") {
      use_stdin = true;
    } else if (arg == "--diagnose-handoff") {
      config.diagnose_handoff = true;
    } else if (arg == "--diagnose-decode-drift") {
      config.diagnose_decode_drift = true;
    } else if (arg == "--dump-step-hiddens" && i + 1 < argc) {
      config.dump_step_hiddens = std::stoi(argv[++i]);
    } else if (arg == "--dump-step-components" && i + 1 < argc) {
      config.dump_step_components = std::stoi(argv[++i]);
    } else if (arg == "--experiment-attn-o-proj-f32-residual") {
      config.experiment_attn_o_proj_f32_residual = true;
    } else if (arg == "--experiment-mlp-down-f32-residual") {
      config.experiment_mlp_down_f32_residual = true;
    } else if (arg == "--help") {
      std::cout << "usage: spock-decode --repack-dir DIR [options]\n";
      std::cout << "  --prompt TEXT       Prompt text\n";
      std::cout << "  --tokens FILE       File with space-separated token IDs\n";
      std::cout << "  --stdin             Read prompt from stdin\n";
      std::cout << "  --max-new-tokens N  Tokens to generate (default 16)\n";
      std::cout << "  --verbose / -v      Verbose output\n";
      std::cout << "  --debug-dump        Dump hidden state after each layer\n";
      std::cout << "  --diagnose-handoff  After prefill, dump chunk vs recurrent state comparison\n";
      std::cout << "  --diagnose-decode-drift  After decode, compare free-run vs rebuilt state at step 5\n";
      std::cout << "  --dump-step-hiddens N   Dump per-layer hiddens at decode step N (stderr JSON)\n";
      std::cout << "  --dump-step-components N Dump component-level intermediates at decode step N (stderr JSON)\n";
      std::cout << "  --experiment-attn-o-proj-f32-residual  Diagnostic fp32 attention o_proj->residual path\n";
      std::cout << "  --experiment-mlp-down-f32-residual  Diagnostic fp32 MLP down_proj->residual path\n";
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
    std::string item;
    while (f >> item) {
      for (const char c : item) {
        if (c < '0' || c > '9') {
          std::cerr << "spock-decode: token file must contain only whitespace-separated unsigned integer token IDs\n";
          return 2;
        }
      }
      std::uint64_t parsed = 0;
      std::istringstream token_stream(item);
      token_stream >> parsed;
      if (!token_stream || parsed > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "spock-decode: invalid token ID: " << item << "\n";
        return 2;
      }
      config.prompt_tokens.push_back(static_cast<std::uint32_t>(parsed));
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

  if (config.prompt_tokens.empty() && !config.prompt_text.empty()) {
    std::cerr << "spock-decode: --prompt/--stdin text tokenization is not implemented; use --tokens FILE\n";
    return 2;
  }

  if (!config.diagnose_handoff) {
    std::cout << "{\n";
    std::cout << "  \"repack_dir\": \"" << config.repack_dir << "\",\n";
    std::cout << "  \"max_new_tokens\": " << config.max_new_tokens << ",\n";
  }
  auto result = spock::runtime::run_vk_decode(config);

  if (!result.error.empty()) {
    if (!config.diagnose_handoff) {
      std::cout << "  \"error\": \"" << result.error << "\"\n";
      std::cout << "}\n";
    }
    return 3;
  }

  if (!config.diagnose_handoff) {
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
    std::cout << "  \"prefill_ms\": " << result.prefill_ms << ",\n";
    std::cout << "  \"decode_ms\": " << result.decode_ms << ",\n";
    std::cout << "  \"decode_submit_count\": " << result.decode_submit_count << ",\n";
    std::cout << "  \"chunked_decode_submit_count\": " << result.chunked_decode_submit_count << ",\n";
    if (result.gpu_decode_us > 0.0) {
      std::cout << "  \"gpu_decode_us\": " << result.gpu_decode_us << ",\n";
    }
    if (!result.per_token_ms.empty()) {
      std::cout << "  \"per_token_ms\": [";
      for (size_t i = 0; i < result.per_token_ms.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result.per_token_ms[i];
      }
      std::cout << "],\n";
    }
    if (!result.per_token_gpu_us.empty()) {
      std::cout << "  \"per_token_gpu_us\": [";
      for (size_t i = 0; i < result.per_token_gpu_us.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result.per_token_gpu_us[i];
      }
      std::cout << "],\n";
    }
    if (!result.gpu_region_us.empty()) {
      std::cout << "  \"gpu_region_us\": {";
      bool first = true;
      for (const auto& [name, us] : result.gpu_region_us) {
        if (!first) std::cout << ",";
        std::cout << "\n    \"" << name << "\": " << us;
        first = false;
      }
      std::cout << "\n  },\n";
    }
    std::cout << "  \"status\": \"ok\"\n";
    std::cout << "}\n";
  }

  return 0;
}
