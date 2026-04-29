#include "runtime/deltanet_chunk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace spock::runtime {

namespace {

std::size_t qkv_index(std::size_t head, std::size_t token, std::size_t dim,
                      std::size_t sequence_length, std::size_t width) {
  return ((head * sequence_length) + token) * width + dim;
}

std::size_t scalar_index(std::size_t head, std::size_t token, std::size_t sequence_length) {
  return head * sequence_length + token;
}

std::size_t state_index(std::size_t head, std::size_t key, std::size_t value,
                        std::size_t key_dim, std::size_t value_dim) {
  return ((head * key_dim) + key) * value_dim + value;
}

void l2_norm_token(std::vector<float>& values, std::size_t head, std::size_t token,
                   std::size_t sequence_length, std::size_t width) {
  constexpr float kEps = 1e-6f;
  float sum_squares = 0.0f;
  for (std::size_t dim = 0; dim < width; ++dim) {
    const float x = values[qkv_index(head, token, dim, sequence_length, width)];
    sum_squares += x * x;
  }
  const float inv_norm = 1.0f / std::sqrt(sum_squares + kEps);
  for (std::size_t dim = 0; dim < width; ++dim) {
    values[qkv_index(head, token, dim, sequence_length, width)] *= inv_norm;
  }
}

std::vector<float> checked_copy(const std::vector<float>& values, std::size_t expected,
                                const char* name) {
  if (values.size() != expected) {
    throw std::runtime_error(std::string(name) + " size mismatch: expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(values.size()));
  }
  return values;
}

}  // namespace

DeltaNetChunkOutputs run_deltanet_chunk_rule(const DeltaNetChunkConfig& config,
                                             const DeltaNetChunkInputs& inputs) {
  if (config.num_heads == 0 || config.sequence_length == 0 || config.key_dim == 0 ||
      config.value_dim == 0 || config.chunk_size == 0) {
    throw std::runtime_error("DeltaNet chunk config fields must be non-zero");
  }

  const std::size_t heads = config.num_heads;
  const std::size_t seq = config.sequence_length;
  const std::size_t k_dim = config.key_dim;
  const std::size_t v_dim = config.value_dim;
  const std::size_t chunk = config.chunk_size;

  std::vector<float> query = checked_copy(inputs.query, heads * seq * k_dim, "query");
  std::vector<float> key = checked_copy(inputs.key, heads * seq * k_dim, "key");
  std::vector<float> value = checked_copy(inputs.value, heads * seq * v_dim, "value");
  std::vector<float> g = checked_copy(inputs.g, heads * seq, "g");
  std::vector<float> beta = checked_copy(inputs.beta, heads * seq, "beta");
  std::vector<float> state;
  if (inputs.initial_state.empty()) {
    state.assign(heads * k_dim * v_dim, 0.0f);
  } else {
    state = checked_copy(inputs.initial_state, heads * k_dim * v_dim, "initial_state");
  }

  if (config.use_qk_l2norm) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t token = 0; token < seq; ++token) {
        l2_norm_token(query, head, token, seq, k_dim);
        l2_norm_token(key, head, token, seq, k_dim);
      }
    }
  }

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));
  for (float& x : query) x *= q_scale;

  const std::size_t pad = (chunk - (seq % chunk)) % chunk;
  const std::size_t total_seq = seq + pad;
  std::vector<float> query_padded(heads * total_seq * k_dim, 0.0f);
  std::vector<float> key_padded(heads * total_seq * k_dim, 0.0f);
  std::vector<float> value_padded(heads * total_seq * v_dim, 0.0f);
  std::vector<float> g_padded(heads * total_seq, 0.0f);
  std::vector<float> beta_padded(heads * total_seq, 0.0f);

  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < seq; ++token) {
      for (std::size_t dim = 0; dim < k_dim; ++dim) {
        query_padded[qkv_index(head, token, dim, total_seq, k_dim)] =
            query[qkv_index(head, token, dim, seq, k_dim)];
        key_padded[qkv_index(head, token, dim, total_seq, k_dim)] =
            key[qkv_index(head, token, dim, seq, k_dim)];
      }
      for (std::size_t dim = 0; dim < v_dim; ++dim) {
        value_padded[qkv_index(head, token, dim, total_seq, v_dim)] =
            value[qkv_index(head, token, dim, seq, v_dim)];
      }
      g_padded[scalar_index(head, token, total_seq)] = g[scalar_index(head, token, seq)];
      beta_padded[scalar_index(head, token, total_seq)] = beta[scalar_index(head, token, seq)];
    }
  }

  std::vector<float> core_attn_out(heads * total_seq * v_dim, 0.0f);
  const std::size_t chunk_count = total_seq / chunk;

  std::vector<float> decay(chunk * chunk, 0.0f);
  std::vector<float> attn(chunk * chunk, 0.0f);
  std::vector<float> solved_value(chunk * v_dim, 0.0f);
  std::vector<float> k_cumdecay(chunk * k_dim, 0.0f);
  std::vector<float> q_chunk(chunk * k_dim, 0.0f);
  std::vector<float> k_chunk(chunk * k_dim, 0.0f);
  std::vector<float> v_chunk(chunk * v_dim, 0.0f);
  std::vector<float> gcum(chunk, 0.0f);
  std::vector<float> beta_chunk(chunk, 0.0f);
  std::vector<float> weighted_state(k_dim * v_dim, 0.0f);
  std::vector<float> local_attn(chunk * chunk, 0.0f);
  std::vector<float> v_prime(chunk * v_dim, 0.0f);
  std::vector<float> v_new(chunk * v_dim, 0.0f);
  std::vector<float> attn_inter(chunk * v_dim, 0.0f);
  std::vector<float> update_state(k_dim * v_dim, 0.0f);

  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
      const std::size_t token_base = chunk_idx * chunk;
      float running_g = 0.0f;
      for (std::size_t t = 0; t < chunk; ++t) {
        const std::size_t token = token_base + t;
        gcum[t] = running_g + g_padded[scalar_index(head, token, total_seq)];
        running_g = gcum[t];
        beta_chunk[t] = beta_padded[scalar_index(head, token, total_seq)];

        for (std::size_t dim = 0; dim < k_dim; ++dim) {
          q_chunk[t * k_dim + dim] = query_padded[qkv_index(head, token, dim, total_seq, k_dim)];
          k_chunk[t * k_dim + dim] = key_padded[qkv_index(head, token, dim, total_seq, k_dim)];
        }
        for (std::size_t dim = 0; dim < v_dim; ++dim) {
          v_chunk[t * v_dim + dim] = value_padded[qkv_index(head, token, dim, total_seq, v_dim)];
        }
      }

      std::fill(decay.begin(), decay.end(), 0.0f);
      std::fill(attn.begin(), attn.end(), 0.0f);
      for (std::size_t row = 0; row < chunk; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
          decay[row * chunk + col] = std::exp(gcum[row] - gcum[col]);
        }
      }

      for (std::size_t row = 0; row < chunk; ++row) {
        for (std::size_t col = 0; col < row; ++col) {
          float dot = 0.0f;
          for (std::size_t dim = 0; dim < k_dim; ++dim) {
            dot += (k_chunk[row * k_dim + dim] * beta_chunk[row]) *
                   k_chunk[col * k_dim + dim];
          }
          attn[row * chunk + col] = -dot * decay[row * chunk + col];
        }
      }
      for (std::size_t row = 1; row < chunk; ++row) {
        std::vector<float> old_row(row, 0.0f);
        for (std::size_t col = 0; col < row; ++col) old_row[col] = attn[row * chunk + col];
        for (std::size_t col = 0; col < row; ++col) {
          float sum = old_row[col];
          for (std::size_t inner = 0; inner < row; ++inner) {
            sum += old_row[inner] * attn[inner * chunk + col];
          }
          attn[row * chunk + col] = sum;
        }
      }
      for (std::size_t diag = 0; diag < chunk; ++diag) {
        attn[diag * chunk + diag] += 1.0f;
      }

      std::fill(solved_value.begin(), solved_value.end(), 0.0f);
      std::fill(k_cumdecay.begin(), k_cumdecay.end(), 0.0f);
      for (std::size_t row = 0; row < chunk; ++row) {
        for (std::size_t col = 0; col < chunk; ++col) {
          const float coeff = attn[row * chunk + col];
          if (coeff == 0.0f) continue;
          const float beta_col = beta_chunk[col];
          const float exp_g_col = std::exp(gcum[col]);
          for (std::size_t vd = 0; vd < v_dim; ++vd) {
            solved_value[row * v_dim + vd] += coeff * (v_chunk[col * v_dim + vd] * beta_col);
          }
          for (std::size_t kd = 0; kd < k_dim; ++kd) {
            k_cumdecay[row * k_dim + kd] +=
                coeff * (k_chunk[col * k_dim + kd] * beta_col * exp_g_col);
          }
        }
      }

      std::fill(local_attn.begin(), local_attn.end(), 0.0f);
      for (std::size_t row = 0; row < chunk; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
          float dot = 0.0f;
          for (std::size_t dim = 0; dim < k_dim; ++dim) {
            dot += q_chunk[row * k_dim + dim] * k_chunk[col * k_dim + dim];
          }
          local_attn[row * chunk + col] = dot * decay[row * chunk + col];
        }
      }

      std::fill(v_prime.begin(), v_prime.end(), 0.0f);
      std::fill(v_new.begin(), v_new.end(), 0.0f);
      std::fill(attn_inter.begin(), attn_inter.end(), 0.0f);
      for (std::size_t row = 0; row < chunk; ++row) {
        const float exp_g_row = std::exp(gcum[row]);
        for (std::size_t vd = 0; vd < v_dim; ++vd) {
          float carry = 0.0f;
          float inter = 0.0f;
          for (std::size_t kd = 0; kd < k_dim; ++kd) {
            carry += k_cumdecay[row * k_dim + kd] *
                     state[state_index(head, kd, vd, k_dim, v_dim)];
            inter += (q_chunk[row * k_dim + kd] * exp_g_row) *
                     state[state_index(head, kd, vd, k_dim, v_dim)];
          }
          v_prime[row * v_dim + vd] = carry;
          v_new[row * v_dim + vd] = solved_value[row * v_dim + vd] - carry;
          attn_inter[row * v_dim + vd] = inter;
        }
      }

      for (std::size_t row = 0; row < chunk; ++row) {
        const std::size_t token = token_base + row;
        for (std::size_t vd = 0; vd < v_dim; ++vd) {
          float sum = attn_inter[row * v_dim + vd];
          for (std::size_t col = 0; col <= row; ++col) {
            sum += local_attn[row * chunk + col] * v_new[col * v_dim + vd];
          }
          core_attn_out[qkv_index(head, token, vd, total_seq, v_dim)] = sum;
        }
      }

      const float exp_g_last = std::exp(gcum[chunk - 1]);
      for (std::size_t kd = 0; kd < k_dim; ++kd) {
        for (std::size_t vd = 0; vd < v_dim; ++vd) {
          weighted_state[kd * v_dim + vd] =
              state[state_index(head, kd, vd, k_dim, v_dim)] * exp_g_last;
        }
      }
      std::fill(update_state.begin(), update_state.end(), 0.0f);
      for (std::size_t row = 0; row < chunk; ++row) {
        const float weight = std::exp(gcum[chunk - 1] - gcum[row]);
        for (std::size_t kd = 0; kd < k_dim; ++kd) {
          const float scaled_k = k_chunk[row * k_dim + kd] * weight;
          for (std::size_t vd = 0; vd < v_dim; ++vd) {
            update_state[kd * v_dim + vd] += scaled_k * v_new[row * v_dim + vd];
          }
        }
      }
      for (std::size_t kd = 0; kd < k_dim; ++kd) {
        for (std::size_t vd = 0; vd < v_dim; ++vd) {
          state[state_index(head, kd, vd, k_dim, v_dim)] =
              weighted_state[kd * v_dim + vd] + update_state[kd * v_dim + vd];
        }
      }
    }
  }

  DeltaNetChunkOutputs outputs;
  outputs.core_attn_out.resize(heads * seq * v_dim);
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < seq; ++token) {
      for (std::size_t vd = 0; vd < v_dim; ++vd) {
        outputs.core_attn_out[qkv_index(head, token, vd, seq, v_dim)] =
            core_attn_out[qkv_index(head, token, vd, total_seq, v_dim)];
      }
    }
  }
  outputs.final_state = std::move(state);
  return outputs;
}

}  // namespace spock::runtime
