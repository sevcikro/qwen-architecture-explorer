"""Look inside a real Qwen3 forward pass: shapes, the next-token distribution,
hidden states of every block and one attention map.

Run:  python 13_hf_inspect.py      (needs: torch, transformers; downloads Qwen3-0.6B, about 1.5 GB)
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("QWEN_MODEL_DIR") or "Qwen/Qwen3-0.6B"
REV = None if os.path.isdir(MODEL) else "c1899de289a04d12100db370d81485cdf75e47ca"
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32, attn_implementation="eager")
model.eval()

print(model.model.layers[0])                     # one decoder block: attention + MLP + two RMSNorms

ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids, output_hidden_states=True, output_attentions=True)

print("\ninput ids      :", ids.tolist()[0])
print("logits shape   :", tuple(out.logits.shape), "= [batch, positions, vocabulary rows]")
p = out.logits[0, -1].softmax(-1)
top = p.topk(5)
for prob, i in zip(top.values, top.indices):
    print(f"   next token {tok.decode([int(i)])!r:<10} p = {prob:.3f}")

hs = out.hidden_states                           # embedding, blocks 0..26, then the final norm output
print("\nhidden states  :", len(hs), "tensors of shape", tuple(hs[0].shape))
names = {0: "embedding", 1: "block 0", 14: "block 13", 27: "block 26", 28: "final norm"}
for l, name in names.items():
    print(f"   after {name:<10} RMS of last position = {hs[l][0, -1].pow(2).mean().sqrt():.3f}")

att = out.attentions[13][0, 0]                   # block 13, head 0: [query, key]
print("\nattention block 13 head 0 (rows sum to 1, upper triangle is 0 = causal):")
print(att.numpy().round(3))
