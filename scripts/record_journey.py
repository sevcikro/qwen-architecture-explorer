"""Record a real chat turn of Qwen3-0.6B for the course's "message journey" simulation.

For the message "What is a parameter?" (enable_thinking=False, the same prompt as the
tokenizer example), run greedy decoding with a K/V cache and store, for every step,
the chosen token and the five most probable candidates with their probabilities.

Usage:
  python -I scripts/record_journey.py --model-dir PATH/TO/Qwen3-0.6B --out public/data/journey-qwen3-0.6b.json
"""
import argparse
import datetime
import hashlib
import json
import pathlib

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REV = "c1899de289a04d12100db370d81485cdf75e47ca"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    md = pathlib.Path(a.model_dir)
    sha = hashlib.sha256((md / "model.safetensors").read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32)
    model.eval()
    messages = [{"role": "user", "content": "What is a parameter?"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").input_ids
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    steps = []
    with torch.no_grad():
        out = model(ids, use_cache=True)                       # prefill: the whole prompt at once
        cache = out.past_key_values
        logits = out.logits[0, -1]
        for _ in range(a.max_new):
            p = logits.float().softmax(-1)
            top = p.topk(5)
            nxt = int(top.indices[0])                          # greedy: the most probable token
            steps.append({
                "id": nxt,
                "piece": tok.decode([nxt]),
                "p": round(float(top.values[0]), 4),
                "top": [{"piece": tok.decode([int(i)]), "p": round(float(v), 4)} for v, i in zip(top.values, top.indices)],
            })
            if nxt == end_id:
                break
            out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)   # decode: one token
            cache = out.past_key_values
            logits = out.logits[0, -1]
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "revision": REV,
        "modelSha256": sha,
        "library": f"transformers {transformers.__version__}, torch {torch.__version__}",
        "dtype": "float32",
        "decoding": "greedy (argmax), K/V cache",
        "recorded": datetime.date.today().isoformat(),
        "messages": messages,
        "renderedPrompt": prompt,
        "promptIds": ids[0].tolist(),
        "promptPieces": [tok.decode([i]) for i in ids[0].tolist()],
        "steps": steps,
        "stoppedOnEnd": bool(steps and steps[-1]["id"] == end_id),
        "answer": tok.decode([s["id"] for s in steps], skip_special_tokens=True).strip(),
    }
    pathlib.Path(a.out).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("prompt tokens:", len(data["promptIds"]), " generated:", len(steps), " stopped on <|im_end|>:", data["stoppedOnEnd"])
    print("answer:", data["answer"])


if __name__ == "__main__":
    main()
