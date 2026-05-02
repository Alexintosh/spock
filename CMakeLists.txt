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

foreach(app spock-bench spock-check vk-capabilities vk_barrier_probe spock-repack-text spock-decode spock-deltanet-chunk spock-deltanet-chunk-prefill-probe spock-deltanet-prefill-collect-probe spock-deltanet-prefill-pipeline-probe spock-deltanet-chunk-prefill-tiled-probe)
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
  add_test(NAME spock_capabilities COMMAND vk-capabilities)
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
  endif()
endif()
