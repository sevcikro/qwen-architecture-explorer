"""The real Qwen3 tokenizer: text -> token ids -> text, and the chat template.

Run:  python 04_tokenizer.py      (needs: transformers; downloads only the tokenizer files)
Set QWEN_MODEL_DIR to a local copy of Qwen/Qwen3-0.6B to work offline.
"""
import os
from transformers import AutoTokenizer

MODEL = os.environ.get("QWEN_MODEL_DIR") or "Qwen/Qwen3-0.6B"
REV = None if os.path.isdir(MODEL) else "c1899de289a04d12100db370d81485cdf75e47ca"  # pinned revision
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)

text = "Hlavné mesto Slovenska je Bratislava."
ids = tok.encode(text)
print("ids   :", ids)
print("pieces:", [tok.decode([i]) for i in ids])
print("count :", len(ids), "tokens for", len(text), "characters")
print("back  :", tok.decode(ids))

# A chat message is first wrapped in the model's chat template (special tokens):
messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("\nchat template output:")
print(prompt)
print("special tokens used:", [t for t in tok.tokenize(prompt) if t.startswith("<")])
print("vocabulary size (tokenizer):", len(tok))
