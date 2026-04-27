#include <cmath>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Task {
  std::string role_path;
  std::string name;
  std::string file;
  std::uint64_t offset = 0;
  std::uint64_t nbytes = 0;
  std::string source_dtype;
  std::string output_dtype;
  std::string shape;
};

std::vector<std::string> split_tsv(const std::string& line) {
  std::vector<std::string> out;
  std::string current;
  std::istringstream stream(line);
  while (std::getline(stream, current, '\t')) out.push_back(current);
  return out;
}

std::vector<Task> read_tasks(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("unable to open task file: " + path.string());
  std::string line;
  std::getline(input, line);
  std::vector<Task> tasks;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    auto fields = split_tsv(line);
    if (fields.size() != 8) throw std::runtime_error("invalid task row: " + line);
    tasks.push_back(Task{
        .role_path = fields[0],
        .name = fields[1],
        .file = fields[2],
        .offset = static_cast<std::uint64_t>(std::stoull(fields[3])),
        .nbytes = static_cast<std::uint64_t>(std::stoull(fields[4])),
        .source_dtype = fields[5],
        .output_dtype = fields[6],
        .shape = fields[7],
    });
  }
  return tasks;
}

std::uint64_t align_to(const std::uint64_t value, const std::uint64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

std::uint16_t fp32_to_fp16_bits(const float value) {
  union {
    float f;
    std::uint32_t u;
  } in{value};
  const std::uint32_t bits = in.u;
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFU;

  if ((bits & 0x7FFFFFFFU) == 0) return sign;
  if (((bits >> 23) & 0xFFU) == 0xFFU) {
    if (mantissa != 0) return static_cast<std::uint16_t>(sign | 0x7E00U);
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000U;
    const int shift = 14 - exponent;
    std::uint16_t half = static_cast<std::uint16_t>(mantissa >> shift);
    if ((mantissa >> (shift - 1)) & 1U) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
  std::uint16_t half = static_cast<std::uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
  if (mantissa & 0x1000U) ++half;
  return half;
}

void convert_bf16_to_fp16(std::vector<char>& bytes) {
  if (bytes.size() % 2 != 0) throw std::runtime_error("BF16 task has odd byte size");
  for (std::size_t i = 0; i < bytes.size(); i += 2) {
    const auto bf16 = static_cast<std::uint16_t>(static_cast<unsigned char>(bytes[i])) |
                      static_cast<std::uint16_t>(static_cast<unsigned char>(bytes[i + 1]) << 8);
    union {
      std::uint32_t u;
      float f;
    } expanded{static_cast<std::uint32_t>(bf16) << 16};
    const std::uint16_t fp16 = fp32_to_fp16_bits(expanded.f);
    bytes[i] = static_cast<char>(fp16 & 0xFFU);
    bytes[i + 1] = static_cast<char>((fp16 >> 8) & 0xFFU);
  }
}

std::string fnv1a64_file_hex(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("unable to hash output file");
  std::uint64_t hash = 1469598103934665603ULL;
  char buffer[1024 * 1024];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const auto count = input.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

void write_json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch == '\n') out << "\\n";
    else out << ch;
  }
  out << '"';
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path tasks_path;
  std::filesystem::path source_root;
  std::filesystem::path output_dir;
  std::uint64_t alignment = 256;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tasks" && i + 1 < argc) tasks_path = argv[++i];
    else if (arg == "--source-root" && i + 1 < argc) source_root = argv[++i];
    else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
    else if (arg == "--alignment" && i + 1 < argc) alignment = std::stoull(argv[++i]);
    else if (arg == "--help") {
      std::cout << "usage: spock-repack-text --tasks TASKS.tsv --source-root ARTIFACT_DIR --output-dir OUT [--alignment N]\n";
      return 0;
    }
  }

  try {
    if (tasks_path.empty() || source_root.empty() || output_dir.empty()) {
      throw std::runtime_error("--tasks, --source-root, and --output-dir are required");
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw std::runtime_error("--alignment must be a power of two");
    }

    const auto tasks = read_tasks(tasks_path);
    std::filesystem::create_directories(output_dir);
    const auto weights_path = output_dir / "text_weights.bin";
    const auto manifest_path = output_dir / "text_repack_manifest.json";

    std::ofstream weights(weights_path, std::ios::binary | std::ios::trunc);
    if (!weights) throw std::runtime_error("unable to create " + weights_path.string());

    struct OutputTensor {
      Task task;
      std::uint64_t offset;
    };
    std::vector<OutputTensor> output_tensors;
    std::uint64_t offset = 0;
    std::vector<char> padding(alignment, '\0');
    for (const auto& task : tasks) {
      const auto aligned = align_to(offset, alignment);
      if (aligned > offset) {
        weights.write(padding.data(), static_cast<std::streamsize>(aligned - offset));
        offset = aligned;
      }

      std::ifstream source(source_root / task.file, std::ios::binary);
      if (!source) throw std::runtime_error("unable to open source file: " + (source_root / task.file).string());
      source.seekg(static_cast<std::streamoff>(task.offset));
      std::vector<char> bytes(static_cast<std::size_t>(task.nbytes));
      source.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (source.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("short read for " + task.role_path);
      }
      if (task.source_dtype == "bf16" && task.output_dtype == "fp16") {
        convert_bf16_to_fp16(bytes);
      } else if (!(task.source_dtype == "f32" && task.output_dtype == "fp32")) {
        throw std::runtime_error("unsupported dtype conversion for " + task.role_path);
      }
      weights.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      output_tensors.push_back(OutputTensor{task, offset});
      offset += task.nbytes;
    }
    weights.close();

    const auto file_size = std::filesystem::file_size(weights_path);
    std::ofstream manifest(manifest_path, std::ios::trunc);
    manifest << "{\n";
    manifest << "  \"schema_version\": 1,\n";
    manifest << "  \"artifact\": {\"format\": \"spock-text-repacked-weights\", \"source_plan\": \"native-task-export\"},\n";
    manifest << "  \"packing\": {\"alignment\": " << alignment
             << ", \"byte_order\": \"little\", \"weights_file\": \"text_weights.bin\"},\n";
    manifest << "  \"files\": [{\"path\": \"text_weights.bin\", \"size_bytes\": " << file_size
             << ", \"fnv1a64\": \"" << fnv1a64_file_hex(weights_path) << "\"}],\n";
    manifest << "  \"tensors\": [\n";
    for (std::size_t i = 0; i < output_tensors.size(); ++i) {
      const auto& item = output_tensors[i];
      manifest << "    {\"role_path\": ";
      write_json_string(manifest, item.task.role_path);
      manifest << ", \"name\": ";
      write_json_string(manifest, item.task.name);
      manifest << ", \"file\": \"text_weights.bin\", \"offset\": " << item.offset
               << ", \"nbytes\": " << item.task.nbytes << ", \"dtype\": ";
      write_json_string(manifest, item.task.output_dtype);
      manifest << ", \"shape\": " << item.task.shape << ", \"source_dtype\": ";
      write_json_string(manifest, item.task.source_dtype);
      manifest << ", \"source_file\": ";
      write_json_string(manifest, (source_root / item.task.file).string());
      manifest << ", \"source_offset\": " << item.task.offset
               << ", \"source_nbytes\": " << item.task.nbytes << "}";
      manifest << (i + 1 == output_tensors.size() ? "\n" : ",\n");
    }
    const auto fp32_count = std::count_if(output_tensors.begin(), output_tensors.end(), [](const auto& item) {
      return item.task.output_dtype == "fp32";
    });
    manifest << "  ],\n";
    manifest << "  \"summary\": {\"tensor_count\": " << output_tensors.size()
             << ", \"fp16_tensors\": " << (output_tensors.size() - static_cast<std::size_t>(fp32_count))
             << ", \"fp32_tensors\": " << fp32_count << ", \"size_bytes\": " << file_size << "}\n";
    manifest << "}\n";
    std::cerr << "wrote native repack manifest: " << manifest_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "spock-repack-text: " << error.what() << '\n';
    return 2;
  }
}
