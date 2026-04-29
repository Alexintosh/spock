#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def flatten(tensor):
    return tensor.reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description="Validate native DeltaNet chunk rule")
    parser.add_argument("--runner", required=True, help="Path to spock-deltanet-chunk executable")
    args = parser.parse_args()

    torch.manual_seed(0)
    num_heads = 2
    sequence_length = 5
    key_dim = 3
    value_dim = 2
    chunk_size = 4

    query = torch.randn(1, sequence_length, num_heads, key_dim, dtype=torch.float32)
    key = torch.randn(1, sequence_length, num_heads, key_dim, dtype=torch.float32)
    value = torch.randn(1, sequence_length, num_heads, value_dim, dtype=torch.float32)
    g = torch.randn(1, sequence_length, num_heads, dtype=torch.float32) * 0.2 - 0.3
    beta = torch.sigmoid(torch.randn(1, sequence_length, num_heads, dtype=torch.float32))
    initial_state = torch.randn(1, num_heads, key_dim, value_dim, dtype=torch.float32) * 0.1

    expected_out, expected_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    fixture = {
        "num_heads": num_heads,
        "sequence_length": sequence_length,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunk_size": chunk_size,
        "use_qk_l2norm": True,
        "query": flatten(query.squeeze(0).transpose(0, 1).contiguous()),
        "key": flatten(key.squeeze(0).transpose(0, 1).contiguous()),
        "value": flatten(value.squeeze(0).transpose(0, 1).contiguous()),
        "initial_state": flatten(initial_state.squeeze(0)),
    }
    # Reorder g/beta to [head][token].
    fixture["g"] = flatten(g.squeeze(0).transpose(0, 1).contiguous())
    fixture["beta"] = flatten(beta.squeeze(0).transpose(0, 1).contiguous())

    with tempfile.TemporaryDirectory(prefix="spock-deltanet-chunk-") as tmpdir:
        input_path = Path(tmpdir) / "fixture.json"
        input_path.write_text(json.dumps(fixture), encoding="utf-8")
        proc = subprocess.run(
            [args.runner, "--input", str(input_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        actual = json.loads(proc.stdout)

    expected_out = expected_out.squeeze(0).transpose(0, 1).contiguous()
    expected_state = expected_state.squeeze(0).contiguous()

    actual_out = torch.tensor(actual["core_attn_out"], dtype=torch.float32).reshape(expected_out.shape)
    actual_state = torch.tensor(actual["final_state"], dtype=torch.float32).reshape(expected_state.shape)

    out_diff = torch.max(torch.abs(actual_out - expected_out)).item()
    state_diff = torch.max(torch.abs(actual_state - expected_state)).item()

    summary = {
        "max_core_attn_out_diff": out_diff,
        "max_final_state_diff": state_diff,
    }
    print(json.dumps(summary, indent=2))

    tolerance = 1e-5
    if out_diff > tolerance or state_diff > tolerance:
        print("FAIL: native chunk rule diverged from torch reference", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
