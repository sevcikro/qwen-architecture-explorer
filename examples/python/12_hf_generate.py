"""Chat with a real Qwen3 model through Hugging Face Transformers: the full path from a
message to an answer (chat template -> token ids -> prefill + decode loop -> text).

Run:  python 12_hf_generate.py      (needs: torch, transformers; downloads Qwen3-0.6B, about 1.5 GB)
Set QWEN_MODEL_DIR to a local copy to work offline.
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("QWEN_MODEL_DIR") or "Qwen/Qwen3-0.6B"
REV = None if os.path.isdir(MODEL) else "c1899de289a04d12100db370d81485cdf75e47ca"
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32)
model.eval()

# 1) the message, wrapped by the chat template
messages = [{"role": "user", "content": "Explain in one sentence what a neural network weight is."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt")
print("prompt tokens:", inputs.input_ids.shape[1])

# 2) prefill + decode loop (greedy, so the result is reproducible)
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
dt = time.perf_counter() - t0
new = out[0, inputs.input_ids.shape[1]:]

# 3) token ids back to text
print("answer:", tok.decode(new, skip_special_tokens=True).strip())
print(f"generated {len(new)} tokens; stopped on {'<|im_end|>' if new[-1].item() == tok.convert_tokens_to_ids('<|im_end|>') else 'the length limit'}")
print(f"(timing depends entirely on your hardware: {dt:.1f} s here)")
