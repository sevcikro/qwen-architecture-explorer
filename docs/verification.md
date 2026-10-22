# Verification report

## Automated

| Check | Result |
|---|---|
| TypeScript `tsc --noEmit` (strict) | 0 errors |
| Unit tests (`npx vitest run`) | 57 / 57 passed |
| Browser tests (`npx playwright test`, Chromium; desktop 1366×860 and Pixel 7) against the production build | 33 passed, 3 skipped (device-specific by design) |
| Same browser suite against the live here.now URL | see `deploy/DEPLOYMENT.md` |

### What the unit tests establish

- Totals 8,190,735,360 and 30,532,122,624 from the shape table **and** from an independent hand-written formula **and** equal to the element totals of the pinned safetensors headers (and index `total_size` / 2).
- Every logical tensor family matches a real checkpoint family in shape, count, element total and dtype; no biases; untied; q/k norms counted once per block; 30B-A3B maps 2048 → 4096 → 2048.
- Card "non-embedding" figures (6.95B / 29.9B) reproduced only by subtracting both vocabulary matrices.
- Packed-expert mapping counts scalars once.
- RMSNorm, SiLU, SwiGLU, stable softmax (masked → 0, all-masked policy, NaN rejected), RoPE (half-split pairing, norm preservation, relative-offset invariance) against hand fixtures; brief example y = [−0.5, 0.4].
- Official-size router fixture: 50 trials of 128 logits → 8 unique experts, α sums to 1, selection is the true top-8; toy top-2 separately.
- KV: 147,456 / 98,304 B per token; 4.5 / 3 GiB at 32,768 tokens.
- Toy engine (dense and MoE): determinism per seed; causal rows normalized with exact zeros in the future; future-token edits leave earlier logits unchanged; cached decoding = full recomputation (≤ 1e-9) incl. RoPE offsets and [Hkv, T, dh] cache shapes; GQA heads sharing K/V produce different outputs; prefix edits truncate, weight edits clear, decoding afterwards matches recomputation; unselected-expert edits leave outputs unchanged; MoE output = α-weighted sum.
- Sampling: T = 0 greedy; top-k → top-p → renormalize; seeded draws reproducible and non-mutating.
- Training toy: autograd loss = independent forward; every parameter tensor passes central finite differences; SGD step equals w − lr·g; large steps can raise loss (measured).
- Hybrid toys: delta-rule recall, gate range, QSA block budget + tail.
- Data: every real sample re-hashes to its SHA-256 from its stored values, matches shard/dtype/shape in the header table, stays under budget; tokenizer arithmetic 151,643 + 26 = 151,669, 267 padding rows.

### What the browser tests establish

Start-with-one-weight shows the real value and file offset; top-down drill from model to scalar with breadcrumbs; keyboard-only navigation (focus, arrows, Enter) into MoE and an expert; model switching translates or clears selections with a notice; deep links survive reload and invalid ones are corrected with an explanation; zooming reveals extra node detail; attention, cache (equivalence + invalidation), router, weight (explicit edit + reset), sampling (RNG only advances on draw), training (finite-difference ✓, recomputed loss), memory (KV figures), parameters, virtualized catalog (aria-rowcount 18,867 with < 40 rows rendered), hybrid/pipeline/sources pages; no uncaught errors, console errors or failed requests on the checked pages; no horizontal page scroll on 10 routes at both sizes; phone menu and bottom-sheet inspector.

## Manual inspection

- Rendered pages reviewed in screenshots (desktop dark, phone light): home, explorer MoE module, attention, router, training, hybrid, parameters. Fixes applied: nav wrapping, phone toolbar height, treemap label contrast, scroll containment.
- Independent byte check: 2 bytes at the computed file offset of Qwen3-8B `q_proj[0,0]` decode to the displayed value.

## Scientific audit (separate from visual QA)

Checked across the app: model-family separation (hybrid facts appear only on hybrid pages); parameter vs neuron vs activation wording; attention probabilities vs W_Q/K/V/O; router parameters vs mixture coefficients; no biases anywhere; `[out, in]` orientation and Xᵀ conventions; head_dim not derived from d/Hq; context terms kept distinct; toy vs real labels (every toy view carries "Illustrative simulation"; real values only where a checksummed sample covers the coordinate); "active" counts presented as conventions. Corrections made during the audit: non-embedding convention, tour percentage (computed instead of hard-coded), removal of an unsupported MTP speculative-decoding claim, norm_topk_prob default resolved from the implementation.

## Performance (measured, not claimed)

Production bundle: main chunk 620 kB (192 kB gzip), lazily loaded page chunks, data files loaded on demand (30B offsets table 1.5 MB only when a 30B coordinate is inspected). The memory lab offers an in-browser measurement of the toy simulator that reports the user agent and CPU count it ran on.

## Remaining limitations

See README "Known limitations" and `docs/scientific-notes.md`.
