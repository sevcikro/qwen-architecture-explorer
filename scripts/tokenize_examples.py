#!/usr/bin/env python3
"""Produce ACTUAL Qwen3 tokenizer outputs for a few fixed example prompts.

Uses the pinned Qwen/Qwen3-8B tokenizer.json with the Hugging Face `tokenizers`
library, and renders the pinned chat_template from tokenizer_config.json with
Jinja2 using the same environment options as transformers' chat-template
renderer (trim_blocks, lstrip_blocks, loopcontrols, sandboxed).

These are tokenizer outputs only. No model is executed, so no activations,
logits or generated text are produced here.

Usage:
  python -I scripts/tokenize_examples.py --tok-dir DIR_WITH_tokenizer.json_AND_tokenizer_config.json --out OUT.json
Requires: tokenizers==0.22.1, jinja2 (install in an isolated virtual environment).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

EXAMPLES = [
    {"id": "raw-capital", "kind": "raw", "text": "The capital of France is"},
    {"id": "raw-weights", "kind": "raw", "text": "Where are the weights stored?"},
    {"id": "raw-czech", "kind": "raw", "text": "Kde jsou uloženy váhy modelu?"},
    {
        "id": "chat-nothink",
        "kind": "chat",
        "messages": [{"role": "user", "content": "What is a parameter?"}],
        "enable_thinking": False,
    },
    {
        "id": "chat-think",
        "kind": "chat",
        "messages": [{"role": "user", "content": "What is a parameter?"}],
        "enable_thinking": True,
    },
]


def render(template: str, messages, enable_thinking: bool) -> str:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking, tools=None
    )


def main() -> int:
    from tokenizers import Tokenizer, __version__ as tv

    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.tok_dir)
    tj = (d / "tokenizer.json").read_bytes()
    cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_str(tj.decode("utf-8"))
    added = {int(k): v["content"] for k, v in cfg["added_tokens_decoder"].items()}
    base_vocab = tok.get_vocab_size(with_added_tokens=False)
    full_vocab = tok.get_vocab_size(with_added_tokens=True)
    results = []
    for ex in EXAMPLES:
        if ex["kind"] == "raw":
            text = ex["text"]
        else:
            text = render(cfg["chat_template"], ex["messages"], ex["enable_thinking"])
        enc = tok.encode(text, add_special_tokens=False)
        pieces = [tok.decode([i], skip_special_tokens=False) for i in enc.ids]
        results.append(
            {
                **{k: v for k, v in ex.items()},
                "renderedText": text,
                "ids": enc.ids,
                "tokens": enc.tokens,
                "decodedPieces": pieces,
                "special": [i in added for i in enc.ids],
            }
        )
    out = {
        "evidence": "actual-tokenizer-output",
        "model": REPO,
        "revision": REVISION,
        "tokenizerJsonSha256": hashlib.sha256(tj).hexdigest(),
        "library": f"tokenizers {tv}",
        "chatTemplateSource": "tokenizer_config.json chat_template (pinned revision)",
        "generated": _dt.date.today().isoformat(),
        "baseVocabSize": base_vocab,
        "vocabWithAddedTokens": full_vocab,
        "addedTokens": [{"id": k, "content": v, "special": cfg["added_tokens_decoder"][str(k)]["special"]} for k, v in sorted(added.items())],
        "note": "Tokenizer output only. No Qwen model was run to produce these files.",
        "examples": results,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"base={base_vocab} with_added={full_vocab} added={len(added)}")
    for r in results:
        print(r["id"], len(r["ids"]), r["ids"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
