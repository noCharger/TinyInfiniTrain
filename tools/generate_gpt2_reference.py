#!/usr/bin/env python3
"""Generate a deterministic FP16 TinyGPT-2 one-step PyTorch reference.

The binary contains the exact initial parameters, token batch, logits before
the update, and logits after an FP32-master Adam update. The matching C++ test
loads the same parameters, so this checks implementation parity rather than
random-number-generator parity.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is required only to regenerate the parity fixture; install it in this Python environment first"
    ) from error


MAGIC = 0x54475032
VERSION = 1


def specs(block: int, vocab: int, layers: int, embd: int):
    result: dict[str, tuple[int, ...]] = {
        "transformer.wte.weight": (vocab, embd),
        "transformer.wpe.weight": (block, embd),
    }
    for layer in range(layers):
        prefix = f"transformer.h.{layer}"
        result.update(
            {
                f"{prefix}.ln_1.weight": (embd,),
                f"{prefix}.ln_1.bias": (embd,),
                f"{prefix}.attn.c_attn.weight": (3 * embd, embd),
                f"{prefix}.attn.c_attn.bias": (3 * embd,),
                f"{prefix}.attn.c_proj.weight": (embd, embd),
                f"{prefix}.attn.c_proj.bias": (embd,),
                f"{prefix}.ln_2.weight": (embd,),
                f"{prefix}.ln_2.bias": (embd,),
                f"{prefix}.mlp.c_fc.weight": (4 * embd, embd),
                f"{prefix}.mlp.c_fc.bias": (4 * embd,),
                f"{prefix}.mlp.c_proj.weight": (embd, 4 * embd),
                f"{prefix}.mlp.c_proj.bias": (embd,),
            }
        )
    result["transformer.ln_f.weight"] = (embd,)
    result["transformer.ln_f.bias"] = (embd,)
    result["lm_head.weight"] = (vocab, embd)
    result["lm_head.bias"] = (vocab,)
    return result


def deterministic_values(name: str, shape: tuple[int, ...], device: torch.device):
    canonical = "transformer.wte.weight" if name == "lm_head.weight" else name
    if canonical.endswith("ln_1.weight") or canonical.endswith("ln_2.weight") or canonical == "transformer.ln_f.weight":
        return torch.ones(shape, dtype=torch.float16, device=device)
    if canonical.endswith("bias"):
        return torch.zeros(shape, dtype=torch.float16, device=device)
    digest = hashlib.sha256(canonical.encode()).digest()
    frequency = 1 + int.from_bytes(digest[:2], "little") % 97
    count = math.prod(shape)
    index = torch.arange(1, count + 1, dtype=torch.float32, device=device)
    values = 0.02 * torch.sin(index * (0.007 * frequency))
    return values.reshape(shape).to(torch.float16)


def build_parameters(config, device):
    values: dict[str, torch.nn.Parameter] = {}
    tied_weight = None
    for name, shape in specs(config.block, config.vocab, config.layers, config.embd).items():
        if name == "lm_head.weight":
            values[name] = tied_weight
            continue
        parameter = torch.nn.Parameter(deterministic_values(name, shape, device))
        values[name] = parameter
        if name == "transformer.wte.weight":
            tied_weight = parameter
    return values


def linear(x, params, prefix):
    return F.linear(x, params[f"{prefix}.weight"], params[f"{prefix}.bias"])


def forward(tokens, params, config):
    batch, sequence = tokens.shape
    positions = torch.arange(sequence, dtype=torch.long, device=tokens.device)
    hidden = F.embedding(tokens, params["transformer.wte.weight"])
    hidden = hidden + F.embedding(positions, params["transformer.wpe.weight"])
    head_size = config.embd // config.heads
    causal = torch.tril(torch.ones((sequence, sequence), dtype=torch.bool, device=tokens.device))
    for layer in range(config.layers):
        prefix = f"transformer.h.{layer}"
        normalized = F.layer_norm(
            hidden,
            (config.embd,),
            params[f"{prefix}.ln_1.weight"],
            params[f"{prefix}.ln_1.bias"],
            1e-5,
        )
        qkv = linear(normalized, params, f"{prefix}.attn.c_attn")
        query, key, value = qkv.split(config.embd, dim=-1)
        query = query.view(batch, sequence, config.heads, head_size).transpose(1, 2)
        key = key.view(batch, sequence, config.heads, head_size).transpose(1, 2)
        value = value.view(batch, sequence, config.heads, head_size).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        attention = attention.masked_fill(~causal.view(1, 1, sequence, sequence), float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        projected = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch, sequence, config.embd)
        hidden = hidden + linear(projected, params, f"{prefix}.attn.c_proj")

        normalized = F.layer_norm(
            hidden,
            (config.embd,),
            params[f"{prefix}.ln_2.weight"],
            params[f"{prefix}.ln_2.bias"],
            1e-5,
        )
        mlp = linear(normalized, params, f"{prefix}.mlp.c_fc")
        mlp = 0.5 * mlp * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (mlp + 0.044715 * mlp.pow(3))))
        hidden = hidden + linear(mlp, params, f"{prefix}.mlp.c_proj")

    hidden = F.layer_norm(
        hidden,
        (config.embd,),
        params["transformer.ln_f.weight"],
        params["transformer.ln_f.bias"],
        1e-5,
    )
    return F.linear(hidden, params["lm_head.weight"], params["lm_head.bias"])


def adam_master_step(parameters, masters, moments, variances, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    with torch.no_grad():
        for parameter in parameters:
            grad = parameter.grad.float()
            key = id(parameter)
            moments[key].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            variances[key].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            corrected_m = moments[key] / (1.0 - beta1**step)
            corrected_v = variances[key] / (1.0 - beta2**step)
            masters[key].addcdiv_(corrected_m, corrected_v.sqrt().add_(eps), value=-lr)
            parameter.copy_(masters[key].to(torch.float16))


def write_fixture(path: Path, config, params, tokens, targets, before, after):
    state = specs(config.block, config.vocab, config.layers, config.embd)
    with path.open("wb") as output:
        output.write(struct.pack("<II7iI", MAGIC, VERSION, config.block, config.vocab, config.layers,
                                 config.heads, config.embd, config.batch, config.sequence, len(state)))
        for name in sorted(state):
            encoded = name.encode()
            data = params[name].detach().float().cpu().contiguous().numpy()
            output.write(struct.pack("<I", len(encoded)))
            output.write(encoded)
            output.write(struct.pack("<Q", data.size))
            output.write(data.tobytes())
        output.write(tokens.cpu().contiguous().numpy().astype("int64").tobytes())
        output.write(targets.cpu().contiguous().numpy().astype("int64").tobytes())
        output.write(before.detach().float().cpu().contiguous().numpy().tobytes())
        output.write(after.detach().float().cpu().contiguous().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("Data/tiny_gpt2_fp16_reference.bin"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--embd", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    args.sequence = args.block
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    params = build_parameters(args, device)
    unique_parameters = list(dict.fromkeys(params.values()))
    tokens = torch.arange(args.batch * args.sequence, device=device, dtype=torch.long).reshape(args.batch, args.sequence)
    tokens.remainder_(args.vocab)
    targets = (tokens + 1).remainder(args.vocab)
    initial_state = {name: parameter.detach().clone() for name, parameter in params.items()}
    before = forward(tokens, params, args)
    loss = F.cross_entropy(before.float().reshape(-1, args.vocab), targets.reshape(-1))
    loss.backward()
    masters = {id(p): p.detach().float().clone() for p in unique_parameters}
    moments = {id(p): torch.zeros_like(masters[id(p)]) for p in unique_parameters}
    variances = {id(p): torch.zeros_like(masters[id(p)]) for p in unique_parameters}
    adam_master_step(unique_parameters, masters, moments, variances, 1)
    with torch.no_grad():
        after = forward(tokens, params, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_fixture(args.output, args, initial_state, tokens, targets, before, after)
    max_change = (after.float() - before.float()).abs().max().item()
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes), loss={loss.item():.7f}, max_logit_change={max_change:.7f}")


if __name__ == "__main__":
    main()
