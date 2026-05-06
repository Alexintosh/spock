cmake_minimum_required(VERSION 3.22)

project(spock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPOCK_ENABLE_VULKAN "Build Vulkan-backed runtime paths" ON)
option(SPOCK_ENABLE_VALIDATION "Enable Vulkan validation layers at runtime" OFF)

find_package(Vulkan QUIET)
find_package(Python3 COMPONENTS Interpreter QUIET)
find_program(GLSLANG_VALIDATOR glslangValidator)

add_library(spock_core
  src/model/qwen35_config.cpp
  src/reference/qwen35_cpu_reference.cpp
  src/runtime/benchmark.cpp
  src/runtime/deltanet_chunk.cpp
  src/runtime/json_parse.cpp
  src/runtime/vk_allocator.cpp
  src/runtime/vk_context.cpp
  src/runtime/vk_device.cpp
  src/runtime/vk_decode.cpp
  src/runtime/vk_session.cpp
  src/runtime/weight_loader.cpp
)

target_include_directories(spock_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(spock_core PUBLIC
  SPOCK_ENABLE_VALIDATION=$<BOOL:${SPOCK_ENABLE_VALIDATION}>
  SPOCK_HAS_VULKAN=$<BOOL:${Vulkan_FOUND}>
)

if(Vulkan_FOUND AND SPOCK_ENABLE_VULKAN)
  target_link_libraries(spock_core PUBLIC Vulkan::Vulkan)
else()
  target_compile_definitions(spock_core PUBLIC SPOCK_VULKAN_STUB=1)
endif()

foreach(app spock-bench spock-check vk-capabilities vk_barrier_probe vk_persistent_decode_skeleton vk_persistent_mlp_probe vk_persistent_layer0_probe vk_residual_add_probe vk_matvec_probe vk_deltanet_norm_gate_probe vk_deltanet_g_beta_probe vk_deltanet_conv_l2_probe vk_deltanet_recurrent_probe vk_deltanet_mixer_probe spock-repack-text spock-decode spock-deltanet-chunk spock-deltanet-chunk-prefill-probe spock-deltanet-prefill-collect-probe spock-deltanet-prefill-pipeline-probe spock-deltanet-chunk-prefill-tiled-probe)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE spock_core)
endforeach()

# Expose compiled shader directory to the decode app
target_compile_definitions(spock_core PUBLIC
  SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

set(SPOCK_SHADER_SOURCES
  shaders/trivial_compute.comp
  shaders/persistent_barrier_probe.comp
  shaders/embedding_lookup.comp
  shaders/embedding_lookup_from_buffer.comp
  shaders/rms_norm.comp
  shaders/matvec.comp
  shaders/matvec_f32_out.comp
  shaders/argmax.comp
  shaders/silu_gate.comp
  shaders/residual_add.comp
  shaders/residual_add_mixed.comp
  shaders/rope_apply.comp
  shaders/attention_decode.comp
  shaders/kv_cache_store.comp
  shaders/sigmoid_gate.comp
  shaders/rms_norm_per_head.comp
  shaders/deltanet_recurrent.comp
  shaders/conv1d_step.comp
  shaders/deltanet_chunk_prefill.comp
  shaders/deltanet_chunk_prefill_tiled.comp
  shaders/deltanet_norm_gate.comp
  shaders/l2_norm_per_head.comp
  shaders/split_q_gate.comp
  shaders/deltanet_compute_g_beta.comp
  shaders/deltanet_prefill_collect.comp
  shaders/deltanet_chunk_last_to_fp16.comp
  shaders/deltanet_conv_l2_qk.comp
  shaders/deltanet_recurrent_gbeta.comp
  shaders/deltanet_recurrent_gbeta_norm_gate.comp
  shaders/lm_head_tiled.comp
  shaders/persistent_decode_skeleton.comp
  shaders/persistent_mlp_probe.comp
  shaders/matvec_tiled.comp
  shaders/persistent_layer0_probe.comp
)

if(GLSLANG_VALIDATOR)
  set(SPOCK_SHADER_OUTPUTS)
  foreach(shader ${SPOCK_SHADER_SOURCES})
    get_filename_component(shader_name ${shader} NAME)
    set(output ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv)
    add_custom_command(
      OUTPUT ${output}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${shader} -o ${output}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
      VERBATIM
    )
    list(APPEND SPOCK_SHADER_OUTPUTS ${output})
  endforeach()
  add_custom_target(spock_shaders ALL DEPENDS ${SPOCK_SHADER_OUTPUTS})
else()
  add_custom_target(spock_shaders
    COMMAND ${CMAKE_COMMAND} -E echo "glslangValidator not found; skipping shader compilation"
  )
endif()

install(TARGETS spock-bench spock-check vk-capabilities vk_barrier_probe)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME spock_check COMMAND spock-check --prompts ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/prompts.jsonl)
  add_test(NAME spock_bench_tg128 COMMAND spock-bench --mode tg128 --json --warmup 0 --runs 1)
  add_test(NAME spock_bench_output COMMAND spock-bench --mode pp520 --csv --warmup 0 --runs 1 --output ${CMAKE_CURRENT_BINARY_DIR}/spock-pp520.csv)
  add_test(NAME spock_barrier_probe_help COMMAND vk_barrier_probe --help)
  add_test(NAME spock_barrier_probe_decode_shape COMMAND vk_barrier_probe --tokens 16 --layers 24 --workgroups 8 --payload-cols 128)
  add_test(NAME spock_barrier_probe_qwen35_preset COMMAND vk_barrier_probe --qwen35-decode-shape-preset)
  add_test(NAME spock_capabilities COMMAND vk-capabilities)
  add_test(NAME spock_persistent_decode_skeleton_help COMMAND vk_persistent_decode_skeleton --help)
  add_test(NAME spock_persistent_decode_skeleton_smoke COMMAND vk_persistent_decode_skeleton --tokens 2 --layers 4 --hidden 128 --workgroups 8)
  add_test(NAME spock_persistent_decode_skeleton_real_weight_smoke COMMAND vk_persistent_decode_skeleton --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.mlp_gate --tokens 1 --layers 1 --hidden 128 --workgroups 4)
  add_test(NAME spock_persistent_decode_skeleton_multi_role_smoke COMMAND vk_persistent_decode_skeleton --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.mlp_gate --weight-role layer.0.mlp_up --tokens 1 --layers 1 --hidden 128 --workgroups 4)
  add_test(NAME spock_persistent_decode_skeleton_row_count_real_weight_smoke COMMAND vk_persistent_decode_skeleton --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.mlp_gate --tokens 1 --layers 1 --hidden 128 --workgroups 4 --row-count 16)
  add_test(NAME spock_persistent_decode_skeleton_row_count_model_width_smoke COMMAND vk_persistent_decode_skeleton --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.mlp_gate --tokens 1 --layers 1 --hidden 1024 --workgroups 82 --row-count 128)
  add_test(NAME spock_residual_add_probe_help COMMAND vk_residual_add_probe --help)
  add_test(NAME spock_residual_add_probe_layer0_mixer_residual_exact COMMAND vk_residual_add_probe --length 1024 --input-a-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_input_hidden_1024.fp16 --input-b-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_output_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16)
  add_test(NAME spock_matvec_probe_help COMMAND vk_matvec_probe --help)
  add_test(NAME spock_matvec_probe_layer0_delta_out_proj_exact COMMAND vk_matvec_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_out_proj --in-dim 2048 --out-dim 1024 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_gated_2048.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_out_1024.fp16)
  add_test(NAME spock_matvec_probe_layer0_delta_z_proj_exact COMMAND vk_matvec_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_in_proj_z --in-dim 1024 --out-dim 2048 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_z_2048.fp16)
  add_test(NAME spock_matvec_probe_layer0_delta_qkv_proj_exact COMMAND vk_matvec_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_in_proj_qkv --in-dim 1024 --out-dim 6144 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_qkv_raw_6144.fp16)
  add_test(NAME spock_matvec_probe_layer0_delta_a_proj_exact COMMAND vk_matvec_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_in_proj_a --in-dim 1024 --out-dim 16 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_a_16.fp16)
  add_test(NAME spock_matvec_probe_layer0_delta_b_proj_exact COMMAND vk_matvec_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_in_proj_b --in-dim 1024 --out-dim 16 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_b_16.fp16)
  add_test(NAME spock_deltanet_norm_gate_probe_help COMMAND vk_deltanet_norm_gate_probe --help)
  add_test(NAME spock_deltanet_norm_gate_probe_layer0_exact COMMAND vk_deltanet_norm_gate_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --weight-role layer.0.delta_norm --num-heads 16 --head-dim 128 --core-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_core_2048.fp16 --gate-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_z_2048.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_gated_2048.fp16)
  add_test(NAME spock_deltanet_g_beta_probe_help COMMAND vk_deltanet_g_beta_probe --help)
  add_test(NAME spock_deltanet_g_beta_probe_layer0_exact COMMAND vk_deltanet_g_beta_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --num-heads 16 --layer-index 0 --a-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_a_16.fp16 --b-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_b_16.fp16 --expected-g-beta-bits-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_g_beta_32.u32)
  add_test(NAME spock_deltanet_conv_l2_probe_help COMMAND vk_deltanet_conv_l2_probe --help)
  add_test(NAME spock_deltanet_conv_l2_probe_layer0_exact COMMAND vk_deltanet_conv_l2_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --raw-qkv-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_qkv_raw_6144.fp16 --conv-state-pre-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 --expected-q-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_q_2048.fp16 --expected-k-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_k_2048.fp16 --expected-v-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_v_2048.fp16)

  add_test(NAME spock_deltanet_recurrent_probe_help COMMAND vk_deltanet_recurrent_probe --help)
  add_test(NAME spock_deltanet_recurrent_probe_layer0_exact COMMAND vk_deltanet_recurrent_probe --q-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_q_2048.fp16 --k-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_k_2048.fp16 --v-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_v_2048.fp16 --g-beta-bits-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_g_beta_32.u32 --state-pre-f32-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_core_2048.fp16)
  add_test(NAME spock_deltanet_mixer_probe_help COMMAND vk_deltanet_mixer_probe --help)
  add_test(NAME spock_deltanet_mixer_probe_layer0_exact COMMAND vk_deltanet_mixer_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --input-norm-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --input-hidden-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_input_hidden_1024.fp16 --conv-state-pre-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 --state-pre-f32-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 --expected-mixer-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_output_1024.fp16 --expected-mixer-residual-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16)
  add_test(NAME spock_persistent_layer0_probe_help COMMAND vk_persistent_layer0_probe --help)
  add_test(NAME spock_persistent_layer0_probe_post_mlp_exact_fails COMMAND vk_persistent_layer0_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16)
  add_test(NAME spock_persistent_layer0_probe_post_mlp_bounded COMMAND vk_persistent_layer0_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 87 --output-fp16-population-ulp-threshold 16 --output-fp16-max-rows-above-population-threshold 10)
  set_tests_properties(spock_persistent_layer0_probe_post_mlp_exact_fails PROPERTIES WILL_FAIL TRUE)

  # Projection-prefix mode: exact fails (8 qkv + 1 z mismatches at 1 ULP, 128-lane reduction-order boundary).
  add_test(NAME spock_persistent_layer0_probe_projection_prefix_exact_fails COMMAND vk_persistent_layer0_probe --mode projections --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --input-norm-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-qkv-raw-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_qkv_raw_6144.fp16 --expected-z-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_z_2048.fp16 --expected-a-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_a_16.fp16 --expected-b-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_b_16.fp16)
  add_test(NAME spock_persistent_layer0_probe_projection_prefix_ulp1 COMMAND vk_persistent_layer0_probe --mode projections --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --input-norm-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_input_norm_1024.fp16 --expected-qkv-raw-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_qkv_raw_6144.fp16 --expected-z-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_z_2048.fp16 --expected-a-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_a_16.fp16 --expected-b-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_b_16.fp16 --projection-fp16-ulp-tolerance 1)
  set_tests_properties(spock_persistent_layer0_probe_projection_prefix_exact_fails PROPERTIES WILL_FAIL TRUE)
  add_test(NAME spock_persistent_layer0_probe_conv_l2_exact COMMAND vk_persistent_layer0_probe --mode conv-l2 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --raw-qkv-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_qkv_raw_6144.fp16 --conv-state-pre-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_conv_state_pre_24576.fp16 --expected-q-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_q_2048.fp16 --expected-k-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_k_2048.fp16 --expected-v-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_v_2048.fp16)
  add_test(NAME spock_persistent_layer0_probe_g_beta_exact COMMAND vk_persistent_layer0_probe --mode g-beta --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --a-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_a_16.fp16 --b-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_b_16.fp16 --expected-g-beta-bits-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_g_beta_32.u32)
  add_test(NAME spock_persistent_layer0_probe_recurrent_exact COMMAND vk_persistent_layer0_probe --mode recurrent --q-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_q_2048.fp16 --k-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_k_2048.fp16 --v-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_v_2048.fp16 --g-beta-bits-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_g_beta_32.u32 --state-pre-f32-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_recurrent_state_pre_262176.f32 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_dn_core_2048.fp16)


  add_test(NAME spock_persistent_mlp_probe_smoke COMMAND vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8)
  add_test(NAME spock_persistent_mlp_probe_real_weight_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8)
  add_test(NAME spock_persistent_mlp_probe_pre_mlp_rmsnorm_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --pre-mlp-rmsnorm)
  add_test(NAME spock_persistent_mlp_probe_full_real_weight_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82)
  add_test(NAME spock_persistent_mlp_probe_full_real_weight_residual_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual)
  add_test(NAME spock_persistent_mlp_probe_full_pre_mlp_rmsnorm_residual_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual)
  add_test(NAME spock_persistent_mlp_probe_full_real_weight_embedding_input_smoke COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-token 0)
  add_test(NAME spock_persistent_mlp_probe_fp16_input_smoke COMMAND vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 8 --workgroups 8 --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mlp_input_pattern_128.fp16)
  add_test(NAME spock_persistent_mlp_probe_captured_fp16_handoff COMMAND vk_persistent_mlp_probe --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer1_captured_fp16_handoff COMMAND vk_persistent_mlp_probe --layer 1 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer1_step1_mixer_residual_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_exact_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_ulp87 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 87)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_population_ulp16 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 87 --output-fp16-population-ulp-threshold 16 --output-fp16-max-rows-above-population-threshold 10)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_population_ulp16_max9_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 87 --output-fp16-population-ulp-threshold 16 --output-fp16-max-rows-above-population-threshold 9)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_down_exact_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_down_output_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_down_ulp2 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_down_output_1024.fp16 --output-fp16-ulp-tolerance 2)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_product_exact_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-mlp-product-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_product_3584.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_product_ulp2 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-mlp-product-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_product_3584.fp16 --output-fp16-ulp-tolerance 2)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_normed_exact COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-norm-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_normed_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_gate_exact_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-gate-scratch-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_gate_3584.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_gate_ulp1 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-gate-scratch-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_gate_3584.fp16 --output-fp16-ulp-tolerance 1)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_up_exact_fails COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-up-scratch-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_up_3584.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer0_captured_rmsnorm_up_ulp2 COMMAND vk_persistent_mlp_probe --layer 0 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mixer_residual_1024.fp16 --expected-up-scratch-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_up_3584.fp16 --output-fp16-ulp-tolerance 2)
  add_test(NAME spock_persistent_mlp_probe_expected_norm_requires_rmsnorm COMMAND vk_persistent_mlp_probe --expected-norm-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer0_step1_mlp_normed_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_expected_output_missing COMMAND vk_persistent_mlp_probe --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/nonexistent_expected_output.fp16)
  add_test(NAME spock_persistent_mlp_probe_expected_output_too_small COMMAND vk_persistent_mlp_probe --hidden 128 --intermediate 16 --output-rows 1024 --workgroups 8 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/mlp_input_pattern_128.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer_invalid_negative COMMAND vk_persistent_mlp_probe --layer -1)
  add_test(NAME spock_persistent_mlp_probe_layer_invalid_string COMMAND vk_persistent_mlp_probe --layer abc)
  add_test(NAME spock_persistent_mlp_probe_layer_invalid_partial COMMAND vk_persistent_mlp_probe --layer 1abc)
  set_tests_properties(spock_persistent_mlp_probe_layer_invalid_partial PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer_invalid_negative PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer_invalid_string PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_population_ulp16_max9_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_down_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_mlp_product_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_gate_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer0_captured_rmsnorm_up_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_expected_norm_requires_rmsnorm PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_expected_output_missing PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_expected_output_too_small PROPERTIES WILL_FAIL TRUE)

  # Layer 20 captured handoff: exact fp16 default MUST fail (2 rows differ by 1 ULP).
  add_test(NAME spock_persistent_mlp_probe_layer20_captured_fp16_handoff_exact_fails COMMAND vk_persistent_mlp_probe --layer 20 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_mixer_residual_1024.fp16)
  set_tests_properties(spock_persistent_mlp_probe_layer20_captured_fp16_handoff_exact_fails PROPERTIES WILL_FAIL TRUE)

  # Layer 20 captured handoff: explicit ULP tolerance 1 must pass.
  add_test(NAME spock_persistent_mlp_probe_layer20_captured_fp16_handoff_ulp1 COMMAND vk_persistent_mlp_probe --layer 20 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_mixer_residual_1024.fp16 --output-fp16-ulp-tolerance 1)
  add_test(NAME spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_exact_fails COMMAND vk_persistent_mlp_probe --layer 20 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_post_mlp_1024.fp16)
  add_test(NAME spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_ulp209_population_ulp16 COMMAND vk_persistent_mlp_probe --layer 20 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 209 --output-fp16-population-ulp-threshold 16 --output-fp16-max-rows-above-population-threshold 1)
  add_test(NAME spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_population_ulp16_max0_fails COMMAND vk_persistent_mlp_probe --layer 20 --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b --hidden 1024 --intermediate 3584 --output-rows 1024 --workgroups 82 --pre-mlp-rmsnorm --residual --input-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_mixer_residual_1024.fp16 --expected-output-fp16-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/layer20_step1_post_mlp_1024.fp16 --output-fp16-ulp-tolerance 209 --output-fp16-population-ulp-threshold 16 --output-fp16-max-rows-above-population-threshold 0)
  set_tests_properties(spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_exact_fails PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_layer20_captured_rmsnorm_mlp_population_ulp16_max0_fails PROPERTIES WILL_FAIL TRUE)

  # Invalid --output-fp16-ulp-tolerance parsing gates.
  add_test(NAME spock_persistent_mlp_probe_ulp_tolerance_invalid_negative COMMAND vk_persistent_mlp_probe --output-fp16-ulp-tolerance -1)
  add_test(NAME spock_persistent_mlp_probe_ulp_tolerance_invalid_string COMMAND vk_persistent_mlp_probe --output-fp16-ulp-tolerance abc)
  add_test(NAME spock_persistent_mlp_probe_ulp_tolerance_invalid_partial COMMAND vk_persistent_mlp_probe --output-fp16-ulp-tolerance 1abc)
  set_tests_properties(spock_persistent_mlp_probe_ulp_tolerance_invalid_negative PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_ulp_tolerance_invalid_string PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(spock_persistent_mlp_probe_ulp_tolerance_invalid_partial PROPERTIES WILL_FAIL TRUE)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME spock_deltanet_chunk_unit
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_deltanet_chunk_unit.py
        --runner $<TARGET_FILE:spock-deltanet-chunk>
    )
    add_test(
      NAME spock_artifact_dry_run
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_artifact_smoke.py
        --converter ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_qwen35_0p8b.py
        --validator ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_artifact.py
    )
    add_test(
      NAME spock_json_fixtures
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_json_fixture_check.py
        --prompts ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/prompts.jsonl
        --baseline ${CMAKE_CURRENT_SOURCE_DIR}/bench/baseline_rx6750xt.json
    )
    add_test(
      NAME spock_diary_check
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_diary_check.py
    )
    add_test(
      NAME spock_chunked_sweep_gpu_timestamp_unit
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_sweep_gpu_timestamp_unit.py
    )
    add_test(
      NAME spock_text_plan_unit
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_text_plan_unit.py
        --planner ${CMAKE_CURRENT_SOURCE_DIR}/tools/plan_text_artifact.py
    )
    add_test(
      NAME spock_p0_parity
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_p0_parity.py
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --check-count 32
    )
    add_test(
      NAME spock_vk_decode_reference_parity
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --limit 8
        --max-new-tokens 16
    )
    add_test(
      NAME spock_vk_decode_prefill_handoff_mismatch
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids mixed_correctness_023,pp520_046
        --max-new-tokens 1
    )
    # Regression gate: GPU-collected chunk-prefill path (double-gated).
    # Non-gated baseline on the same prompt is below.
    add_test(
      NAME spock_vk_decode_gpu_collect_chunk_prefill_short
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 1
    )
    set_tests_properties(spock_vk_decode_gpu_collect_chunk_prefill_short PROPERTIES
      ENVIRONMENT "SPOCK_GPU_CHUNK_PREFILL=1;SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1"
    )
    # Triple-gated tiled-chunk-prefill path (GPU-collect + tiled dispatch).
    add_test(
      NAME spock_vk_decode_gpu_collect_chunk_prefill_tiled
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 1
    )
    set_tests_properties(spock_vk_decode_gpu_collect_chunk_prefill_tiled PROPERTIES
      ENVIRONMENT "SPOCK_GPU_CHUNK_PREFILL=1;SPOCK_GPU_CHUNK_PREFILL_FROM_GPU_COLLECT=1;SPOCK_GPU_CHUNK_PREFILL_TILED=1"
    )
    # Full gated fast Vulkan decode path.
    add_test(
      NAME spock_vk_decode_full_fast_gate_short
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 4
    )
    set_tests_properties(spock_vk_decode_full_fast_gate_short PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1"
    )
    # Chunked-decode gate size-1 equivalence: chunked path enabled with chunk
    # size 1 must produce identical output to the non-chunked single-submit path.
    # Validates that the gate wiring is live without altering decode behavior.
    add_test(
      NAME spock_vk_decode_chunked_gate_size1_fast_gate_short
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 4
    )
    set_tests_properties(spock_vk_decode_chunked_gate_size1_fast_gate_short PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1;SPOCK_GPU_CHUNKED_DECODE=1;SPOCK_GPU_DECODE_CHUNK_SIZE=1"
    )

    # First active bounded chunked-decode path: chunk size 4 with a full
    # eligible chunk followed by a partial final chunk must preserve parity.
    add_test(
      NAME spock_vk_decode_chunked_gate_size4_partial_short
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 6
        --expect-decode-submit-count 2
        --expect-chunked-decode-submit-count 2
    )
    set_tests_properties(spock_vk_decode_chunked_gate_size4_partial_short PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1;SPOCK_GPU_CHUNKED_DECODE=1;SPOCK_GPU_DECODE_CHUNK_SIZE=4"
    )
    # Broader chunked-decode correctness gate: multiple prompts at chunk size 8
    # exercises both full and partial trailing chunks across distinct sequences.
    add_test(
      NAME spock_vk_decode_chunked_gate_size8_multiprompt_16
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001,mixed_correctness_023
        --max-new-tokens 16
        --expect-decode-submit-count 2
        --expect-chunked-decode-submit-count 2
    )
    set_tests_properties(spock_vk_decode_chunked_gate_size8_multiprompt_16 PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1;SPOCK_GPU_CHUNKED_DECODE=1;SPOCK_GPU_DECODE_CHUNK_SIZE=8"
    )
    # With the skip_layers first token recorded into the chunk, chunk size 16
    # covers the full 16-token decode in one command-buffer submit.
    add_test(
      NAME spock_vk_decode_chunked_gate_size16_singlechunk_16
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001,mixed_correctness_023
        --max-new-tokens 16
        --expect-decode-submit-count 1
        --expect-chunked-decode-submit-count 1
    )
    set_tests_properties(spock_vk_decode_chunked_gate_size16_singlechunk_16 PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1;SPOCK_GPU_CHUNKED_DECODE=1;SPOCK_GPU_DECODE_CHUNK_SIZE=16"
    )
    # Basic GPU timestamp coverage for chunked decode. Block timestamps remain
    # disabled under chunked mode; this verifies per-token start/end queries.
    add_test(
      NAME spock_vk_decode_chunked_gate_size16_timestamps_short
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 16
        --expect-decode-submit-count 1
        --expect-chunked-decode-submit-count 1
        --expect-gpu-decode-us-positive
    )
    set_tests_properties(spock_vk_decode_chunked_gate_size16_timestamps_short PROPERTIES
      ENVIRONMENT "SPOCK_GPU_PER_LAYER_DESCRIPTOR_SETS=1;SPOCK_GPU_MERGED_DELTANET=1;SPOCK_GPU_FUSED_DN_CONV_L2=1;SPOCK_GPU_FUSED_DN_GBETA_RECURRENT=1;SPOCK_GPU_FUSED_DN_REC_NORM_GATE=1;SPOCK_GPU_SINGLE_SUBMIT=1;SPOCK_GPU_DEVICE_RESIDENT_TOKEN=1;SPOCK_GPU_DEFER_TOKEN_DOWNLOAD=1;SPOCK_GPU_MATVEC_TILED=1;SPOCK_GPU_LM_HEAD_TILED=1;SPOCK_GPU_CHUNKED_DECODE=1;SPOCK_GPU_DECODE_CHUNK_SIZE=16;SPOCK_GPU_TIMESTAMPS=1"
    )


    # Non-gated baseline on the same prompt for quick diagnostics.
    add_test(
      NAME spock_vk_decode_gpu_collect_chunk_prefill_short_baseline
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_vk_decode_parity.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --ids short_correctness_001
        --max-new-tokens 1
    )

    add_test(
      NAME spock_diagnose_handoff_mc023
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_diagnose_handoff.py
        --decode $<TARGET_FILE:spock-decode>
        --repack-dir ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/spock-text-repack-qwen35-0p8b
        --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference_tokens.jsonl
        --id mixed_correctness_023
    )
    add_test(
      NAME spock_text_repack_unit
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_text_repack_unit.py
        --repacker ${CMAKE_CURRENT_SOURCE_DIR}/tools/repack_text_weights.py
        --native-repacker $<TARGET_FILE:spock-repack-text>
        --validator ${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_text_repack.py
    )
    add_test(
      NAME spock_extract_component_fp16
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_extract_component_fp16_unit.py
    )
  endif()
endif()
