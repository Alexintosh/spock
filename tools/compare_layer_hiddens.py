#!/usr/bin/env python3
"""Compare Vulkan step diagnostics against HF repacked-FP16 execution.

This script is intentionally narrow: it runs the short003-style one-token
sequential HF path and the local Vulkan decoder diagnostics for a selected
decode step, then prints layer hidden and final-RMSNorm differences.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from reference_decode import load_repacked_tensors


DEFAULT_TOKENS = [826, 2250, 5706, 314, 264, 1603, 27502, 11, 1608, 73982, 1132, 13]


def half_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def parse_first_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    return json.loads(text[start:])


def run_vulkan(args: argparse.Namespace, flag: str) -> tuple[dict, dict]:
    with tempfile.NamedTemporaryFile("w", delete=False) as token_file:
        token_file.write(" ".join(str(t) for t in args.tokens))
        token_path = token_file.name

    cmd = [
        args.decode,
        "--repack-dir",
        args.repack_dir,
        "--tokens",
        token_path,
        "--max-new-tokens",
        str(args.max_new_tokens),
        flag,
        str(args.step),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    Path(token_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout), parse_first_json(proc.stderr)


def run_vulkan_debug_top5(args: argparse.Namespace) -> list[tuple[int, float]]:
    with tempfile.NamedTemporaryFile("w", delete=False) as token_file:
        token_file.write(" ".join(str(t) for t in args.tokens))
        token_path = token_file.name

    cmd = [
        args.decode,
        "--repack-dir",
        args.repack_dir,
        "--tokens",
        token_path,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--debug-dump",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    Path(token_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    pattern = re.compile(rf"decode {args.step} top5:(.*)")
    for line in proc.stderr.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        pairs = []
        for token, value in re.findall(r"\((\d+),([-+0-9.eE]+)\)", match.group(1)):
            pairs.append((int(token), float(value)))
        return pairs
    return []


def load_model(model_dir: str, repack_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(torch.float16)
    tensors = load_repacked_tensors(repack_dir)
    state_dict = model.state_dict()
    matched = 0
    for key in list(state_dict):
        if "visual" in key or "mtp" in key:
            continue
        if key in tensors:
            state_dict[key] = tensors[key]
            matched += 1
        elif key == "lm_head.weight" and "model.embed_tokens.weight" in tensors:
            state_dict[key] = tensors["model.embed_tokens.weight"]
            matched += 1
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Injected {matched} repacked tensors from {repack_dir}")
    return model


def tensor_last_token(value) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    return value.detach()[0, -1, :].to(torch.float32).cpu()


def hf_step(args: argparse.Namespace) -> dict:
    model = load_model(args.model_dir, args.repack_dir)

    layer_inputs: dict[int, torch.Tensor] = {}
    mixer_outputs: dict[int, torch.Tensor] = {}
    layer_outputs: dict[int, torch.Tensor] = {}
    mlp_products: dict[int, torch.Tensor] = {}
    down_outputs: dict[int, torch.Tensor] = {}
    dn_input_norms: dict[int, torch.Tensor] = {}
    dn_q: dict[int, torch.Tensor] = {}
    dn_k: dict[int, torch.Tensor] = {}
    dn_v: dict[int, torch.Tensor] = {}
    dn_z: dict[int, torch.Tensor] = {}
    dn_g: dict[int, torch.Tensor] = {}
    dn_beta: dict[int, torch.Tensor] = {}
    dn_core: dict[int, torch.Tensor] = {}
    dn_gated: dict[int, torch.Tensor] = {}
    dn_out: dict[int, torch.Tensor] = {}
    attn_q_norm: dict[int, torch.Tensor] = {}
    attn_k_norm: dict[int, torch.Tensor] = {}
    attn_gate: dict[int, torch.Tensor] = {}
    attn_v: dict[int, torch.Tensor] = {}
    attn_gated: dict[int, torch.Tensor] = {}
    attn_out: dict[int, torch.Tensor] = {}
    final_norm_output: list[torch.Tensor | None] = [None]
    capture: list[bool] = [False]

    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def make_pre_hook(layer_idx: int):
            def hook(_module, inputs):
                if capture[0]:
                    layer_inputs[layer_idx] = tensor_last_token(inputs[0])
            return hook

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                if capture[0]:
                    layer_outputs[layer_idx] = tensor_last_token(output)
            return hook

        def make_mixer_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                if capture[0]:
                    mixer_outputs[layer_idx] = tensor_last_token(output)
            return hook

        hooks.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
        hooks.append(layer.register_forward_hook(make_hook(idx)))
        if hasattr(layer, "input_layernorm"):
            def make_input_norm_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    if capture[0]:
                        dn_input_norms[layer_idx] = tensor_last_token(output)
                return hook

            hooks.append(layer.input_layernorm.register_forward_hook(make_input_norm_hook(idx)))
        if hasattr(layer, "self_attn"):
            hooks.append(layer.self_attn.register_forward_hook(make_mixer_hook(idx)))
            self_attn = layer.self_attn

            def make_attn_q_proj_hook(layer_idx: int, head_dim: int):
                def hook(module, _inputs, output):
                    if not capture[0]:
                        return
                    input_shape = output.shape[:-1]
                    projected = output.view(*input_shape, -1, head_dim * 2)
                    _query, gate = torch.chunk(projected, 2, dim=-1)
                    attn_gate[layer_idx] = gate.reshape(*input_shape, -1).detach()[0, -1].to(torch.float32).cpu()
                return hook

            def make_flat_last_hook(layer_idx: int, target: dict[int, torch.Tensor]):
                def hook(_module, _inputs, output):
                    if capture[0]:
                        target[layer_idx] = output.detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                return hook

            def make_attn_o_pre_hook(layer_idx: int):
                def hook(_module, inputs):
                    if capture[0]:
                        attn_gated[layer_idx] = inputs[0].detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                return hook

            hooks.append(self_attn.q_proj.register_forward_hook(make_attn_q_proj_hook(idx, self_attn.head_dim)))
            hooks.append(self_attn.q_norm.register_forward_hook(make_flat_last_hook(idx, attn_q_norm)))
            hooks.append(self_attn.k_norm.register_forward_hook(make_flat_last_hook(idx, attn_k_norm)))
            hooks.append(self_attn.v_proj.register_forward_hook(make_flat_last_hook(idx, attn_v)))
            hooks.append(self_attn.o_proj.register_forward_pre_hook(make_attn_o_pre_hook(idx)))
            hooks.append(self_attn.o_proj.register_forward_hook(make_flat_last_hook(idx, attn_out)))
        elif hasattr(layer, "linear_attn"):
            hooks.append(layer.linear_attn.register_forward_hook(make_mixer_hook(idx)))
            linear_attn = layer.linear_attn

            def make_dn_norm_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    if capture[0]:
                        dn_gated[layer_idx] = output.detach().reshape(-1).to(torch.float32).cpu()
                return hook

            def make_dn_out_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    if capture[0]:
                        dn_out[layer_idx] = tensor_last_token(output)
                return hook

            def make_dn_z_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    if capture[0]:
                        dn_z[layer_idx] = tensor_last_token(output)
                return hook

            hooks.append(linear_attn.in_proj_z.register_forward_hook(make_dn_z_hook(idx)))
            hooks.append(linear_attn.norm.register_forward_hook(make_dn_norm_hook(idx)))
            hooks.append(linear_attn.out_proj.register_forward_hook(make_dn_out_hook(idx)))

            original_recurrent = linear_attn.recurrent_gated_delta_rule

            def make_recurrent_wrapper(layer_idx: int, original):
                def wrapper(query, key, value, *args, **kwargs):
                    result = original(query, key, value, *args, **kwargs)
                    if capture[0]:
                        q_f32 = query.detach().to(torch.float32)
                        k_f32 = key.detach().to(torch.float32)
                        q_norm = q_f32 * torch.rsqrt((q_f32 * q_f32).sum(dim=-1, keepdim=True) + 1e-6)
                        k_norm = k_f32 * torch.rsqrt((k_f32 * k_f32).sum(dim=-1, keepdim=True) + 1e-6)
                        dn_q[layer_idx] = q_norm[0, -1].reshape(-1).cpu()
                        dn_k[layer_idx] = k_norm[0, -1].reshape(-1).cpu()
                        dn_v[layer_idx] = value.detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                        g = kwargs.get("g")
                        beta = kwargs.get("beta")
                        if g is not None:
                            dn_g[layer_idx] = g.detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                        if beta is not None:
                            dn_beta[layer_idx] = beta.detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                        core = result[0] if isinstance(result, tuple) else result
                        dn_core[layer_idx] = core.detach()[0, -1].reshape(-1).to(torch.float32).cpu()
                    return result

                return wrapper

            linear_attn.recurrent_gated_delta_rule = make_recurrent_wrapper(idx, original_recurrent)

            class RestoreWrapper:
                def remove(self, module=linear_attn, original=original_recurrent):
                    module.recurrent_gated_delta_rule = original

            hooks.append(RestoreWrapper())

        def make_mlp_product_hook(layer_idx: int):
            def hook(module, inputs, _output):
                if capture[0]:
                    x = inputs[0][0, -1, :]  # post_normed hidden, last token
                    gate = module.gate_proj(x)
                    up = module.up_proj(x)
                    product = F.silu(gate) * up
                    mlp_products[layer_idx] = product.detach().to(torch.float32).cpu()
                    down_out = module.down_proj(product)
                    down_outputs[layer_idx] = down_out.detach().to(torch.float32).cpu()
            return hook

        hooks.append(layer.mlp.register_forward_hook(make_mlp_product_hook(idx)))

    def norm_hook(_module, _inputs, output):
        if capture[0]:
            final_norm_output[0] = tensor_last_token(output)

    hooks.append(model.model.norm.register_forward_hook(norm_hook))

    input_ids = torch.tensor([args.tokens], dtype=torch.long)
    generated: list[int] = []
    target_logits: torch.Tensor | None = None

    with torch.no_grad():
        cache = None
        logits = None
        for pos in range(input_ids.shape[1]):
            out = model(input_ids[:, pos : pos + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]

        for step in range(args.max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            # Vulkan diagnostic step N captures the hidden state used to predict
            # generated token N. For N > 0, HF obtains that state by feeding
            # generated token N-1 and reading the resulting logits.
            capture_forward = step + 1 == args.step
            capture[0] = capture_forward
            out = model(next_token, past_key_values=cache, use_cache=True)
            capture[0] = False
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            if capture_forward:
                target_logits = logits[0].detach().to(torch.float32).cpu()

    for hook in hooks:
        hook.remove()

    if target_logits is None or final_norm_output[0] is None:
        raise RuntimeError(f"failed to capture HF step {args.step}")

    return {
        "generated": generated,
        "inputs": layer_inputs,
        "mixer_outputs": mixer_outputs,
        "layers": layer_outputs,
        "mlp_products": mlp_products,
        "down_outputs": down_outputs,
        "dn_input_norms": dn_input_norms,
        "dn_q": dn_q,
        "dn_k": dn_k,
        "dn_v": dn_v,
        "dn_z": dn_z,
        "dn_g": dn_g,
        "dn_beta": dn_beta,
        "dn_core": dn_core,
        "dn_gated": dn_gated,
        "dn_out": dn_out,
        "attn_q_norm": attn_q_norm,
        "attn_k_norm": attn_k_norm,
        "attn_gate": attn_gate,
        "attn_v": attn_v,
        "attn_gated": attn_gated,
        "attn_out": attn_out,
        "final_norm": final_norm_output[0],
        "logits": target_logits,
    }


def vector_stats(vk: np.ndarray, hf: np.ndarray) -> tuple[float, float, float, float, float]:
    diff = np.abs(vk - hf)
    dot = float(np.dot(vk, hf))
    vk_norm = float(np.linalg.norm(vk))
    hf_norm = float(np.linalg.norm(hf))
    cosine = dot / (vk_norm * hf_norm) if vk_norm and hf_norm else 0.0
    return float(diff.max()), float(diff.mean()), cosine, vk_norm, hf_norm


def rms_norm_cpu(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply Qwen3.5 RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight)."""
    x_f32 = x.astype(np.float32)
    w_f32 = weight.astype(np.float32)
    variance = np.mean(x_f32 ** 2)
    rms = np.sqrt(variance + eps)
    return (x_f32 / rms) * (1.0 + w_f32)


def load_repacked_role_np(repack_dir: str, role_path: str) -> np.ndarray:
    manifest_path = Path(repack_dir) / "text_repack_manifest.json"
    weights_path = Path(repack_dir) / "text_weights.bin"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["tensors"]:
        if entry["role_path"] != role_path:
            continue
        with weights_path.open("rb") as handle:
            handle.seek(entry["offset"])
            raw = handle.read(entry["nbytes"])
        if entry["dtype"] == "fp32":
            dtype = np.float32
        elif entry["dtype"] == "fp16":
            dtype = np.float16
        else:
            raise ValueError(f"unsupported dtype {entry['dtype']} for {role_path}")
        return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"]).astype(np.float32)
    raise KeyError(role_path)


def rms_norm_gated_cpu(core: np.ndarray, gate: np.ndarray, weight: np.ndarray, head_dim: int = 128) -> np.ndarray:
    x = core.astype(np.float32).reshape(-1, head_dim)
    z = gate.astype(np.float32).reshape(-1, head_dim)
    w = weight.astype(np.float32).reshape(head_dim)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    normed = x * np.reciprocal(np.sqrt(variance + 1e-6, dtype=np.float32))
    weighted = (w.reshape(1, head_dim) * normed).astype(np.float16).astype(np.float32)
    silu = z / (1.0 + np.exp(-z))
    return (weighted * silu).astype(np.float16).astype(np.float32).reshape(-1)


def print_dn_norm_gate_cpu_checks(vk_components: dict, hf: dict, repack_dir: str, first_layer: int, last_layer: int) -> None:
    print()
    print("DeltaNet norm_gate CPU replay:")
    print("layer  test                              max_abs     mean_abs    cosine      norm_a      norm_b")
    for entry in vk_components.get("layers", []):
        layer = int(entry["layer"])
        if layer < first_layer or layer > last_layer:
            continue
        if not entry.get("dn_core_fp16") or not entry.get("dn_z_fp16") or not entry.get("dn_gated_fp16"):
            continue
        weight = load_repacked_role_np(repack_dir, f"layer.{layer}.delta_norm")
        vk_core = fp16_list_to_np(entry["dn_core_fp16"])
        vk_z = fp16_list_to_np(entry["dn_z_fp16"])
        vk_gated = fp16_list_to_np(entry["dn_gated_fp16"])
        cpu_vk = rms_norm_gated_cpu(vk_core, vk_z, weight)
        max_abs, mean_abs, cosine, na, nb = vector_stats(cpu_vk, vk_gated)
        print(f"{layer:>5}  CPU(VK core,z) vs VK gated       {max_abs:>10.6f} {mean_abs:>10.6f} {cosine:>10.7f} {na:>10.4f} {nb:>10.4f}")

        hf_core = hf["dn_core"].get(layer)
        hf_z = hf["dn_z"].get(layer)
        hf_gated = hf["dn_gated"].get(layer)
        if hf_core is None or hf_z is None or hf_gated is None:
            continue
        cpu_hf = rms_norm_gated_cpu(
            hf_core.numpy().astype(np.float32),
            hf_z.numpy().astype(np.float32),
            weight.astype(np.float16).astype(np.float32),
        )
        hf_gated_vec = hf_gated.numpy().astype(np.float32)
        max_abs, mean_abs, cosine, na, nb = vector_stats(cpu_hf, hf_gated_vec)
        print(f"{layer:>5}  CPU(HF core,z) vs HF gated       {max_abs:>10.6f} {mean_abs:>10.6f} {cosine:>10.7f} {na:>10.4f} {nb:>10.4f}")


def topk(logits: torch.Tensor, k: int = 5) -> list[tuple[int, float]]:
    values, indices = torch.topk(logits, k)
    return [(int(i), float(v)) for v, i in zip(values, indices)]


def fp16_list_to_np(values: list[int]) -> np.ndarray:
    return np.array([half_to_float(int(v)) for v in values], dtype=np.float32)


def print_component_vector_comparison(vk_components: dict, hf: dict, first_layer: int, last_layer: int) -> None:
    print()
    print("Layer component vector comparison:")
    print("layer  component      max_abs      mean_abs     cosine       norm_vk      norm_hf")
    for entry in vk_components.get("layers", []):
        layer = int(entry["layer"])
        if layer < first_layer or layer > last_layer:
            continue
        hf_input = hf["inputs"].get(layer)
        hf_mixer_delta = hf["mixer_outputs"].get(layer)
        hf_post = hf["layers"].get(layer)
        comparisons = []
        if hf_input is not None and entry.get("input_hidden_fp16"):
            comparisons.append(("input", entry["input_hidden_fp16"], hf_input.numpy().astype(np.float32)))
        if hf_input is not None and hf_mixer_delta is not None and entry.get("mixer_residual_fp16"):
            hf_mixer = (hf_input + hf_mixer_delta).numpy().astype(np.float32)
            comparisons.append(("post_mixer", entry["mixer_residual_fp16"], hf_mixer))
        if hf_post is not None and entry.get("post_mlp_fp16"):
            comparisons.append(("post_mlp", entry["post_mlp_fp16"], hf_post.numpy().astype(np.float32)))

        for name, vk_bits, hf_vec in comparisons:
            vk_vec = fp16_list_to_np(vk_bits)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_vec, hf_vec)
            print(f"{layer:>5}  {name:<12}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

        # MLP product comparison (INTER-dimensional, separate entry)
        hf_mlp_prod = hf["mlp_products"].get(layer)
        if hf_mlp_prod is not None and entry.get("mlp_product_fp16"):
            vk_prod = fp16_list_to_np(entry["mlp_product_fp16"])
            hf_prod_vec = hf_mlp_prod.numpy().astype(np.float32)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_prod, hf_prod_vec)
            print(f"{layer:>5}  mlp_product  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

        # Down output comparison (HIDDEN-dimensional, down_proj(mlp_product))
        hf_down = hf["down_outputs"].get(layer)
        if hf_down is not None and entry.get("down_output_fp16"):
            vk_down = fp16_list_to_np(entry["down_output_fp16"])
            hf_down_vec = hf_down.numpy().astype(np.float32)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_down, hf_down_vec)
            print(f"{layer:>5}  down_output  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

        dn_half_components = [
            ("dn_in_norm", "dn_input_norm_fp16", hf["dn_input_norms"].get(layer)),
            ("dn_q", "dn_q_fp16", hf["dn_q"].get(layer)),
            ("dn_k", "dn_k_fp16", hf["dn_k"].get(layer)),
            ("dn_v", "dn_v_fp16", hf["dn_v"].get(layer)),
            ("dn_z", "dn_z_fp16", hf["dn_z"].get(layer)),
            ("dn_core", "dn_core_fp16", hf["dn_core"].get(layer)),
            ("dn_gated", "dn_gated_fp16", hf["dn_gated"].get(layer)),
            ("dn_out", "dn_out_fp16", hf["dn_out"].get(layer)),
        ]
        for name, vk_key, hf_tensor in dn_half_components:
            if hf_tensor is None or not entry.get(vk_key):
                continue
            vk_vec = fp16_list_to_np(entry[vk_key])
            hf_vec = hf_tensor.numpy().astype(np.float32)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_vec, hf_vec)
            print(f"{layer:>5}  {name:<12}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

        dn_float_components = [
            ("dn_g", "dn_g", hf["dn_g"].get(layer)),
            ("dn_beta", "dn_beta", hf["dn_beta"].get(layer)),
        ]
        for name, vk_key, hf_tensor in dn_float_components:
            if hf_tensor is None or not entry.get(vk_key):
                continue
            vk_vec = np.array(entry[vk_key], dtype=np.float32)
            hf_vec = hf_tensor.numpy().astype(np.float32)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_vec, hf_vec)
            print(f"{layer:>5}  {name:<12}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

        attn_half_components = [
            ("attn_q_norm", "attn_q_norm_fp16", hf["attn_q_norm"].get(layer)),
            ("attn_k_norm", "attn_k_norm_fp16", hf["attn_k_norm"].get(layer)),
            ("attn_gate", "attn_gate_fp16", hf["attn_gate"].get(layer)),
            ("attn_v", "attn_v_fp16", hf["attn_v"].get(layer)),
            ("attn_gated", "attn_gated_fp16", hf["attn_gated"].get(layer)),
            ("attn_out", "attn_out_fp16", hf["attn_out"].get(layer)),
        ]
        for name, vk_key, hf_tensor in attn_half_components:
            if hf_tensor is None or not entry.get(vk_key):
                continue
            vk_vec = fp16_list_to_np(entry[vk_key])
            hf_vec = hf_tensor.numpy().astype(np.float32)
            max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_vec, hf_vec)
            print(f"{layer:>5}  {name:<12}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", default="build/spock-decode")
    parser.add_argument("--model-dir", default="artifacts/hf/Qwen--Qwen3.5-0.8B")
    parser.add_argument("--tokenizer-dir", default="artifacts/hf/Qwen--Qwen3.5-0.8B-tokenizer")
    parser.add_argument("--repack-dir", default="artifacts/spock-text-repack-qwen35-0p8b")
    parser.add_argument("--tokens", nargs="+", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=23)
    args = parser.parse_args()

    vk_stdout, vk_hiddens = run_vulkan(args, "--dump-step-hiddens")
    _, vk_components = run_vulkan(args, "--dump-step-components")
    vk_top5 = run_vulkan_debug_top5(args)
    hf = hf_step(args)

    print(f"Vulkan generated: {vk_stdout['generated_tokens']}")
    print(f"HF generated:     {hf['generated']}")
    print(f"HF top5 step {args.step}: {topk(hf['logits'])}")
    print(f"VK top5 step {args.step}: {vk_top5}")
    print_component_vector_comparison(vk_components, hf, args.first_layer, args.last_layer)
    print_dn_norm_gate_cpu_checks(vk_components, hf, args.repack_dir, args.first_layer, args.last_layer)
    print()
    print("Layer hidden comparison (post-layer output):")
    print("layer  max_abs      mean_abs     cosine       norm_vk      norm_hf")
    for entry in vk_hiddens["layers"]:
        layer = int(entry["layer"])
        if layer < args.first_layer or layer > args.last_layer:
            continue
        if layer not in hf["layers"]:
            print(f"{layer:>5}  missing HF hook output")
            continue
        vk_vec = fp16_list_to_np(entry["hidden_fp16"])
        hf_vec = hf["layers"][layer].numpy().astype(np.float32)
        max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_vec, hf_vec)
        print(f"{layer:>5}  {max_abs:>10.6f}  {mean_abs:>10.6f}  {cosine:>10.7f}  {vk_norm:>10.4f}  {hf_norm:>10.4f}")

    vk_final = np.array(
        [half_to_float(int(v)) for v in vk_components["final_norm"]["hidden_fp16"]],
        dtype=np.float32,
    )
    hf_final = hf["final_norm"].numpy().astype(np.float32)
    max_abs, mean_abs, cosine, vk_norm, hf_norm = vector_stats(vk_final, hf_final)
    print()
    print("Final RMSNorm comparison:")
    print(f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} cosine={cosine:.7f} norm_vk={vk_norm:.4f} norm_hf={hf_norm:.4f}")

    # ---- CPU-applied RMSNorm correctness check ----
    repacked = load_repacked_tensors(args.repack_dir)
    norm_weight = repacked["model.norm.weight"].numpy().astype(np.float32)

    vk_layer23_hidden = None
    for entry in vk_hiddens["layers"]:
        if int(entry["layer"]) == 23:
            vk_layer23_hidden = np.array(
                [half_to_float(int(v)) for v in entry["hidden_fp16"]],
                dtype=np.float32,
            )
            break
    hf_layer23_hidden = hf["layers"][23].numpy().astype(np.float32)

    cpu_vk_final = rms_norm_cpu(vk_layer23_hidden, norm_weight)
    cpu_hf_final = rms_norm_cpu(hf_layer23_hidden, norm_weight)

    print()
    print("CPU-applied final RMSNorm comparison:")
    print("test                               max_abs     mean_abs    cosine      norm_a      norm_b")

    max_abs, mean_abs, cosine, na, nb = vector_stats(cpu_vk_final, vk_final)
    print(f"CPU-finalnorm(VK layer23) vs VK dump  {max_abs:>10.6f} {mean_abs:>10.6f} {cosine:>10.7f} {na:>10.4f} {nb:>10.4f}")

    max_abs, mean_abs, cosine, na, nb = vector_stats(cpu_hf_final, hf_final)
    print(f"CPU-finalnorm(HF layer23) vs HF hook  {max_abs:>10.6f} {mean_abs:>10.6f} {cosine:>10.7f} {na:>10.4f} {nb:>10.4f}")

    max_abs, mean_abs, cosine, na, nb = vector_stats(cpu_vk_final, cpu_hf_final)
    print(f"CPU-finalnorm(VK layer23) vs CPU(HF)  {max_abs:>10.6f} {mean_abs:>10.6f} {cosine:>10.7f} {na:>10.4f} {nb:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
