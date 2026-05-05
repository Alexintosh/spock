# Spock Engineering Diary

This diary records the implementation phases of Spock as the project moves from a plan to a working Vulkan-native inference engine.

The intended reader is an experienced programmer who may know little about LLMs, token generation, model artifacts, GPU inference, or Vulkan compute. Entries therefore explain both what changed and why the change matters for inference correctness, reproducibility, and performance.

## Entries

- [0001: Project Skeleton, Parity Contract, and First Build](0001_project_skeleton_and_contract.md)
- [0002: Actual Qwen 3.5 0.8B Artifact Ingestion](0002_actual_model_ingestion.md)
- [0003: Text-Only Artifact Plan](0003_text_only_artifact_plan.md)
- [0004: Text Weight Repack](0004_text_weight_repack.md)
- [0005: Reference Decode P0 Freeze](0005_reference_decode_p0_freeze.md)
- [0006: Weight Pipeline Verification](0006_weight_pipeline_verification.md)
- [0007: Vulkan Decode Pipeline](0007_vulkan_decode_pipeline.md)
- [0008: MLP Decode Path — First Correct GPU Forward Pass](0008_mlp_decode_path.md)
- [0009: Full Attention Layers — KV Cache, QK-Norm, mRoPE, GQA Decode](0009_attention_decode_path.md)
- [0010: DeltaNet Recurrent Decode — All 24 Layers Active](0010_deltanet_decode_path.md)
- [0011: Attention V-Accumulation Bug — First Correct Multi-Layer Output](0011_attention_v_accumulation_bug.md)
- [0012: Qwen3.5 RMSNorm Parity Fix](0012_qwen35_rmsnorm_parity_fix.md)
- [0013: Native DeltaNet Chunk Rule Primitive](0013_native_deltanet_chunk_rule.md)
- [0014: RADV Device Selection and Prefill-Sensitive Hardware Gate](0014_radv_device_selection_and_prefill_gate.md)
- [0015: Decode Drift Diagnostics and Handoff Checkpoint](0015_decode_drift_diagnostics_and_handoff_checkpoint.md)
- [0016: Precision Drift Isolation and Refined Next Work](0016_precision_drift_isolation_and_refined_next_work.md)
- [0017: GPU DeltaNet Chunk-Prefill Shader — First Experimental Probe](0017_gpu_deltanet_chunk_prefill_probe.md)
- [0018: Combined GPU Prefill Pipeline Probe](0018_current_verified_state_checkpoint.md)
- [0019: Runtime Diagnostic GPU Prefill Collection Comparison](0019_gpu_collect_prefill_diagnostic.md)
- [0020: GPU Chunk-Prefill From GPU Collection — Avoiding CPU Intermediate Packing](0020_gpu_chunk_prefill_from_gpu_collect.md)
- [0021: Narrow Runtime Cleanup — Skip CPU Collection Bridge on No-Compare Gated Path](0021_narrow_runtime_cleanup.md)
- [0022: CTest Regression Gate for GPU Collect → GPU Chunk-Prefill Path](0022_ctest_gpu_collect_chunk_prefill_gate.md)
- [0023: Tiled Single-Dispatch Chunk-Prefill Probe — Removing the Per-Head Submit Blocker](0023_tiled_single_dispatch_chunk_prefill_probe.md)
- [0024: Runtime Tiled Chunk-Prefill Gate — Integrated Single-Dispatch Path](0024_runtime_tiled_chunk_prefill_gate.md)
- [0025: GPU-Resident Chunk-Prefill Output Handoff — Removing the CPU Readback/Upload Bridge](0025_gpu_resident_chunk_output_handoff.md)
- [0026: GPU-Side Chunk Init Clear — Removing the CPU Zero-Fill/Staging Bridge](0026_gpu_side_chunk_init_clear.md)
- [0027: Device-Resident Decode Token Embedding — Removing CPU Token Re-Injection](0027_device_resident_decode_token_embedding.md)
- [0028: Deferred Generated-Token Download — Removing Per-Step CPU Token Readback](0028_deferred_generated_token_download.md)
- [0029: Opt-in Per-Layer Stable Descriptor Sets — Reducing Per-Layer Descriptor Mutation Under the Gate](0029_per_layer_stable_descriptor_sets.md)
- [0030: Rejected: Intra-DeltaNet Sub-Step Descriptor Pre-Binding — Decode-State Corruption Under the Gate](0030_rejected_inner_deltanet_descriptor_prebind.md)
- [0031: Pre-bound RoPE Descriptors — Removing the Per-Step RoPE Descriptor Mutation Blocker](0031_prebound_rope_descriptors.md)
- [0032: L2-Norm DeltaNet Descriptor Pre-Binding — Narrow Extension Beyond Diary 0030](0032_l2_deltanet_descriptor_slice.md)
- [0033: Rejected: dn_compute_g_beta Descriptor Pre-Binding — Isolated Failure Beyond Diary 0030](0033_rejected_deltanet_gbeta_descriptor_prebind.md)
- [0034: Corrigendum: dn_compute_g_beta Descriptor Pre-Binding — Constructor Ordering Root Cause Found](0034_deltanet_gbeta_descriptor_prebind_after_ordering_fix.md)
- [0035: dn_recurrent Descriptor Pre-Binding — Final Dispatch-Target Blocker Narrowed](0035_deltanet_recurrent_descriptor_prebind.md)
- [0036: dn_norm_gate Descriptor Pre-Binding — Penultimate Dispatch-Target Blocker](0036_deltanet_norm_gate_descriptor_prebind.md)
- [0037: dn_out_proj Descriptor Pre-Binding — Final Dispatch-Target Blocker Eliminated](0037_deltanet_out_proj_descriptor_prebind.md)
- [0038: Merged DeltaNet Decode Command Buffers — Removing Two Per-Layer Submits Under the Gate](0038_merged_deltanet_decode_command_buffers.md)
- [0039: Single-Submit Decode — One Command Buffer per Decode Token](0039_single_submit_decode.md)
- [0040: GPU Timestamp Decode Instrumentation — Opt-In Measurement Gate](0040_gpu_timestamp_decode_instrumentation.md)
- [0041: Fused DeltaNet Conv+L2 Decode Sub-Block](0041_fused_deltanet_conv_l2_decode.md)
- [0042: Fused DeltaNet G/Beta + Recurrent Decode Sub-Block](0042_fused_deltanet_gbeta_recurrent_decode.md)
- [0043: Fused DeltaNet Recurrent+Norm/Gate Decode Sub-Block](0043_fused_deltanet_recurrent_norm_gate_decode.md)
- [0044: Block-Level GPU Decode Timestamps](0044_block_level_gpu_decode_timestamps.md)
- [0045: Tiled LM-Head Decode Matvec](0045_tiled_lm_head_decode.md)
- [0046: Tiled Decode Matvec](0046_tiled_decode_matvec.md)
- [0047: Persistent Barrier Probe -- Real Vulkan Software Global Barrier](0047_persistent_barrier_probe.md)
- [0048: Persistent Barrier Mini-Pipeline -- Two-Stage Coherent Cross-Read](0048_persistent_barrier_minipipeline.md)
- [0049: Barrier Probe Timestamp Measurement](0049_barrier_probe_timestamps.md)
- [0050: Barrier Probe 1M-Iteration Soak](0050_barrier_probe_1m_soak.md)
- [0051: Barrier Probe Payload Mode](0051_barrier_probe_payload_mode.md)
- [0052: Barrier Probe Memory-Traffic Payload](0052_barrier_probe_memory_payload.md)
- [0053: Memory Payload Soak Boundary](0053_memory_payload_soak_boundary.md)
- [0054: Bounded Memory Payload Repeats](0054_bounded_memory_payload_repeats.md)
- [0055: Barrier Probe In-Process Repeats](0055_barrier_probe_in_process_repeats.md)
- [0056: In-Process 10-Repeat Memory Payload Run](0056_in_process_repeat_10_run.md)
- [0057: Full Fast Decode CTest Gate](0057_full_fast_decode_ctest_gate.md)
- [0058: Chunked Decode Gate Scaffold](0058_chunked_decode_scaffold.md)
- [0059: Chunked Scaffold CTest Gate](0059_chunked_scaffold_ctest_gate.md)
- [0060: Chunked Decode Size-1 Equivalence Gate](0060_chunked_decode_size1_equivalence.md)
- [0061: Bounded Chunked Decode — First Active Multi-Token Path](0061_chunked_decode_size4.md)
- [0062: Submit-Count Instrumentation for Chunked Decode](0062_chunked_submit_count_gate.md)
- [0063: Size-8 Chunked Decode Multiprompt CTest Gate](0063_chunked_decode_size8_multiprompt_gate.md)
- [0064: Chunked Decode Sweep Tool](0064_chunked_decode_sweep_tool.md)
- [0065: Chunked Decode Sweep Repeat/Warmup Extension](0065_chunked_decode_sweep_repeats.md)
- [0066: Chunked Decode Controlled Sweep — Host-Side Timing Evidence](0066_chunked_decode_controlled_sweep.md)
- [0067: Chunked Decode — First skip_layers Decode Step Absorbed Into Chunk](0067_chunked_decode_skip_step_in_chunk.md)
- [0068: Chunked Decode GPU Timestamp Instrumentation](0068_chunked_decode_gpu_timestamps.md)
- [0069: Chunked Decode Sweep GPU Timestamp Extension](0069_chunked_decode_sweep_gpu_timestamps.md)
- [0070: Chunked Decode GPU-Timestamped Controlled Sweep](0070_chunked_decode_gpu_timestamp_sweep.md)
- [0071: Barrier Probe Decode-Shape Gate](0071_barrier_probe_decode_shape_gate.md)
- [0072: Barrier Probe Decode-Shape 82-Workgroup Timing](0072_barrier_probe_decode_shape_82wg_timing.md)
- [0073: Barrier Probe Model-Width Decode-Shaped Payload](0073_barrier_probe_model_width_payload.md)
- [0074: Qwen3.5 Decode-Shape Preset](0074_qwen35_decode_shape_preset.md)
- [0075: Persistent Decode Skeleton -- First fp16/fp32 Persistent Probe](0075_persistent_decode_skeleton.md)
- [0076: Persistent Decode Skeleton -- Real Repacked fp16 Model-Weight Support](0076_persistent_skeleton_real_weight_probe.md)
- [0077: Persistent Decode Skeleton -- Multi-Role Real-Weight Probe](0077_persistent_skeleton_multi_role_probe.md)

## Conventions

Each entry should cover:

- The phase goal.
- The implementation work completed.
- The inference concepts needed to understand the work.
- Verification performed.
- Known limitations and next work.

The diary is not a changelog. It should explain the engineering reasoning behind the code, including mistakes, constraints, and tradeoffs.
