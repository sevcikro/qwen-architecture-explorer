# Changelog

## 1.1.0 — 2026-10-22

- New **Learn** section: 14 chapters that explain the model from a single weight and neuron up to the whole model, text generation, training, memory/hardware and newer hybrid designs. Each chapter: plain-language explanation, everyday comparison, interactive experiment, real Qwen3 numbers (computed from the verified accounting), quick check, links to the explorer and labs, progress tracking.
- 14 new interactive experiments: next-token probabilities, a real checkpoint weight with its BF16 bits, a three-input neuron, the SiLU curve, matrix-vector multiplication, real Qwen3 tokens (now including two Slovak sentences), RMSNorm and the residual addition, attention, a SwiGLU gated channel, a router, a token's vector through the whole toy model, temperature sampling, learning one weight by gradient descent, memory needs.
- **Slovak translation** with a language switch (EN / SK), browser-language detection, remembered choice and shareable `?lang=sk` links. Slovak number formatting (spaces, decimal comma).
- Real tokenizer data now records character offsets, so byte-split characters (e.g. “ď”) are explained instead of shown as “�”.
- Fixes: code font ligatures no longer turn `<|im_start|>` into arrows; header fits one line in both languages and the explorer adapts to the real header height; phone header no longer wraps.
- Tests: 63 unit (new: translation completeness, chapter links and demos) and 41 browser tests (new: course flow, all chapters, demos, Slovak mode).

## 1.0.0 — 2026-10-22

- First release: semantic-zoom explorer (levels 0–6), 12 labs, parameter inventory, tensor catalog, real checkpoint samples, real tokenizer outputs, hybrid-model profiles, tours, sources, glossary.
