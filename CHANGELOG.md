# Changelog

## 1.2.0 — 2026-10-22

- **The course became an interactive book**: 27 chapters in five parts (was 14). New chapters: vectors and dot products; probabilities, softmax and cross-entropy; multi-head attention and GQA; RoPE; build a tiny transformer yourself; prefill, decode and the K/V cache; **from your message to the answer** (a recorded chat turn of Qwen3-0.6B replayed step by step); running and inspecting a real Qwen model; backpropagation step by step; how large language models are trained; serving many users (batching, paged K/V cache, speculative decoding, quantization); inside a checkpoint file; the whole transformer on one page.
- **26 explanatory diagrams** (SVG, bilingual, phone-first) added across the course, **8 new experiments** (dot product, cross-entropy, RoPE rotation, number of K/V heads, cache cost, message journey, backpropagation, quantizing 16 real Qwen3-8B weights) and **16 Python examples** in `examples/python` with their real recorded output (`scripts/run_examples.py`).
- **Real activations**: the *Real run* page (Qwen3-0.6B forward passes with hooks) and a recorded chat turn (`scripts/record_journey.py`), both pinned to revision c1899de2 and checked against the model file hash.
- **More real weights**: 26 checkpoint tiles instead of 7 (every Qwen3-8B tensor family of block 0, q_proj of blocks 17 and 35, the final norm and the output head; more experts, the last router, k_proj and the embedding of Qwen3-30B-A3B). Extraction used 4.12 MB of the 100 MB budget; the 7 earlier tiles are byte-identical.
- **Complete Slovak translation**: all labs, explorer descriptions, catalog, pipeline, family, hybrid, sources (claims and source descriptions), about, error and deep-link messages; the English-only notices are gone.
- **Accessibility**: axe-core audit added to the browser tests (24 routes, both themes, both languages); fixed nested interactive SVG, ARIA grid/table structure, keyboard access to scrollable regions and the light theme’s colour contrast (accent, badges, buttons).
- **Fixes**: on phones the parameter treemap no longer prints clipped labels into thin slices and lists every segment below the map; the About page no longer says that no real activations are shown.
- The GitHub repository is public.

## 1.1.0 — 2026-10-22

- New **Learn** section: 14 chapters that explain the model from a single weight and neuron up to the whole model, text generation, training, memory/hardware and newer hybrid designs. Each chapter: plain-language explanation, everyday comparison, interactive experiment, real Qwen3 numbers (computed from the verified accounting), quick check, links to the explorer and labs, progress tracking.
- 14 new interactive experiments: next-token probabilities, a real checkpoint weight with its BF16 bits, a three-input neuron, the SiLU curve, matrix-vector multiplication, real Qwen3 tokens (now including two Slovak sentences), RMSNorm and the residual addition, attention, a SwiGLU gated channel, a router, a token's vector through the whole toy model, temperature sampling, learning one weight by gradient descent, memory needs.
- **Slovak translation** with a language switch (EN / SK), browser-language detection, remembered choice and shareable `?lang=sk` links. Slovak number formatting (spaces, decimal comma).
- Real tokenizer data now records character offsets, so byte-split characters (e.g. “ď”) are explained instead of shown as “�”.
- Fixes: code font ligatures no longer turn `<|im_start|>` into arrows; header fits one line in both languages and the explorer adapts to the real header height; phone header no longer wraps.
- Tests: 63 unit (new: translation completeness, chapter links and demos) and 41 browser tests (new: course flow, all chapters, demos, Slovak mode).

## 1.0.0 — 2026-10-22

- First release: semantic-zoom explorer (levels 0–6), 12 labs, parameter inventory, tensor catalog, real checkpoint samples, real tokenizer outputs, hybrid-model profiles, tours, sources, glossary.
