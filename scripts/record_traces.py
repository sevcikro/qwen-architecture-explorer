#!/usr/bin/env python3
"""Record REAL activation traces of Qwen/Qwen3-0.6B for Qwen Atlas.

This runs the actual pinned checkpoint on the CPU with Hugging Face transformers and records
a small, bounded subset of what it computes, with full provenance. Run it in an isolated
virtual environment (torch CPU + transformers), never from inside the downloaded model folder:

  python -I scripts/record_traces.py --model-dir DIR_WITH_PINNED_FILES --out public/data/trace-qwen3-0.6b.json

What is recorded per prompt (all from forward hooks on the real modules):
  * token ids and pieces (real tokenizer, pinned chat template)
  * residual-stream RMS at every block output, for every position
  * the first 32 residual features after every block, for the last position
  * attention probabilities (eager attention) of heads 0, 1, 8, 15 in blocks 0, 13, 27
  * the SwiGLU gate activation SiLU(x W_gate^T) of the first 32 channels in blocks 0, 13, 27 (last position)
  * top-10 next-token probabilities at the last position
Floats are rounded to 6 significant digits to keep the file small (stated in the output).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import struct
import sys
import time
from pathlib import Path

REPO = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ATTN_BLOCKS = [0, 13, 27]
ATTN_HEADS = [0, 1, 8, 15]
FEATURES = 32
PROMPTS = [
    {"id": "raw-capital", "kind": "raw", "text": "The capital of France is"},
    {"id": "raw-slovak", "kind": "raw", "text": "Hlavné mesto Slovenska je"},
    {"id": "chat-parameter", "kind": "chat", "messages": [{"role": "user", "content": "What is a parameter in a neural network?"}], "enable_thinking": False},
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_elements(p: Path) -> dict:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    total = 0
    for v in hdr.values():
        e = 1
        for d in v["shape"]:
            e *= d
        total += e
    return {"tensors": len(hdr), "elements": total, "dtypes": sorted({v["dtype"] for v in hdr.values()}), "has_lm_head": "lm_head.weight" in hdr}


def r6(x: float) -> float:
    return float(f"{x:.6g}")


def main() -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    md = Path(a.model_dir)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    load_s = time.time() - t0
    cfg = model.config
    layers = model.model.layers
    L = len(layers)

    cap: dict = {}

    def hook_block(i):
        def f(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap.setdefault("block_out", {})[i] = h.detach()[0]
        return f

    def hook_attn(i):
        def f(_m, _inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                cap.setdefault("attn", {})[i] = out[1].detach()[0]
        return f

    def hook_gate(i):
        def f(_m, inp, _out):
            x = inp[0].detach()[0]
            mlp = layers[i].mlp
            cap.setdefault("gate", {})[i] = torch.nn.functional.silu(x @ mlp.gate_proj.weight.T)
        return f

    hooks = [model.model.embed_tokens.register_forward_hook(lambda _m, _i, o: cap.__setitem__("embed", o.detach()[0]))]
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(hook_block(i)))
        if i in ATTN_BLOCKS:
            hooks.append(layer.self_attn.register_forward_hook(hook_attn(i)))
            hooks.append(layer.mlp.register_forward_hook(hook_gate(i)))
    hooks.append(model.model.norm.register_forward_hook(lambda _m, _i, o: cap.__setitem__("final_norm", o.detach()[0])))

    results = []
    run_s = 0.0
    for pr in PROMPTS:
        cap.clear()
        if pr["kind"] == "raw":
            text = pr["text"]
        else:
            text = tok.apply_chat_template(pr["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=pr["enable_thinking"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]
        t1 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        run_s += time.time() - t1
        T = ids.shape[1]
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 10)
        rms = lambda v: float(torch.sqrt((v.float() ** 2).mean()))
        res = {
            **{k: v for k, v in pr.items()},
            "renderedText": text,
            "ids": ids[0].tolist(),
            "pieces": [tok.decode([i]) for i in ids[0].tolist()],
            "residualRms": {
                "embedding": [r6(rms(cap["embed"][t])) for t in range(T)],
                "blocks": [[r6(rms(cap["block_out"][i][t])) for t in range(T)] for i in range(L)],
                "finalNorm": [r6(rms(cap["final_norm"][t])) for t in range(T)],
            },
            "residualFeaturesLastPos": {
                "features": list(range(FEATURES)),
                "embedding": [r6(x) for x in cap["embed"][-1][:FEATURES].tolist()],
                "blocks": [[r6(x) for x in cap["block_out"][i][-1][:FEATURES].tolist()] for i in range(L)],
            },
            "attention": [
                {"block": b, "head": h, "kvHead": h // (cfg.num_attention_heads // cfg.num_key_value_heads), "probs": [[r6(x) for x in row] for row in cap["attn"][b][h].tolist()]}
                for b in ATTN_BLOCKS for h in ATTN_HEADS
            ],
            "gateLastPos": [{"block": b, "channels": list(range(FEATURES)), "silu": [r6(x) for x in cap["gate"][b][-1][:FEATURES].tolist()]} for b in ATTN_BLOCKS],
            "topNext": [{"id": int(i), "piece": tok.decode([int(i)]), "p": r6(float(p)), "logit": r6(float(logits[int(i)]))} for p, i in zip(top.values, top.indices)],
        }
        results.append(res)
        print(f"{pr['id']}: T={T} top1={res['topNext'][0]}", file=sys.stderr)
    for hk in hooks:
        hk.remove()

    files = {f: sha256(md / f) for f in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]}
    out = {
        "evidence": "measured-trace",
        "model": REPO,
        "revision": REVISION,
        "fileSha256": files,
        "checkpointHeader": header_elements(md / "model.safetensors"),
        "config": {**{k: getattr(cfg, k, None) for k in ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "tie_word_embeddings", "rms_norm_eps"]}, "rope_theta": (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", getattr(cfg, "rope_theta", None))},
        "runtime": {
            "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(), "threads": torch.get_num_threads(),
            "precision": "float32 compute; weights stored as BF16 in the checkpoint and upcast exactly to float32 on load",
            "attnImplementation": "eager (needed to return attention probabilities)", "useCache": False,
            "loadSeconds": round(load_s, 2), "forwardSeconds": round(run_s, 3),
        },
        "hooks": {
            "embedding": "model.model.embed_tokens output",
            "blocks": "output of model.model.layers[i] (residual stream after the block)",
            "attention": f"second output of model.model.layers[i].self_attn (eager softmax probabilities) for blocks {ATTN_BLOCKS}, heads {ATTN_HEADS}",
            "gate": f"SiLU(x @ mlp.gate_proj.weight.T) recomputed from the hooked MLP input, blocks {ATTN_BLOCKS}, first {FEATURES} channels",
            "finalNorm": "model.model.norm output",
        },
        "rounding": "floats rounded to 6 significant digits",
        "recorded": dt.date.today().isoformat(),
        "license": "Apache-2.0 (model repository license)",
        "note": "Real activations of Qwen3-0.6B for the prompts below. Qwen3-0.6B is a smaller model of the same Qwen3 dense family; these are NOT activations of Qwen3-8B or Qwen3-30B-A3B.",
        "prompts": results,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes); load {load_s:.1f}s forward {run_s:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
