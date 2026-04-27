#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::uint32_t iterations = 10000;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--iterations" && i + 1 < argc) iterations = std::stoul(argv[++i]);
    if (arg == "--help") {
      std::cout << "usage: vk_barrier_probe [--iterations N]\n";
      return 0;
    }
  }

  std::cout << "{\n";
  std::cout << "  \"iterations\": " << iterations << ",\n";
  std::cout << "  \"status\": \"not-run\",\n";
  std::cout << "  \"reason\": \"software global barrier probe scaffold; Vulkan stress implementation pending\"\n";
  std::cout << "}\n";
}
