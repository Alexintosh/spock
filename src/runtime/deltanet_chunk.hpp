#pragma once

#include <cstddef>
#include <vector>

namespace spock::runtime {

struct DeltaNetChunkConfig {
  std::size_t num_heads = 0;
  std::size_t sequence_length = 0;
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t chunk_size = 64;
  bool use_qk_l2norm = true;
};

struct DeltaNetChunkInputs {
  // Flattened as [head][token][dim].
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;

  // Flattened as [head][token].
  std::vector<float> g;
  std::vector<float> beta;

  // Optional initial recurrent state, flattened as [head][key_dim][value_dim].
  std::vector<float> initial_state;
};

struct DeltaNetChunkOutputs {
  // Flattened as [head][token][value_dim].
  std::vector<float> core_attn_out;

  // Flattened as [head][key_dim][value_dim].
  std::vector<float> final_state;
};

DeltaNetChunkOutputs run_deltanet_chunk_rule(const DeltaNetChunkConfig& config,
                                             const DeltaNetChunkInputs& inputs);

}  // namespace spock::runtime
