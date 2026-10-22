# Scientific notes

All facts below are pinned to: Qwen/Qwen3-8B `b968826d9c46…`, Qwen/Qwen3-30B-A3B `ad44e777bcd1…`, Qwen/Qwen3.8-Flash-Next `de4b8e4d43b9…`, transformers `17c6736cb2d3…` (retrieved 2026-10-22). Full registry: `src/evidence/sources.ts`, claims: `src/evidence/claims.ts`.

## Conventions

- Activations carry the feature axis last. Stored linear weights are PyTorch `[out, in]`; a projection is `Y = X Wᵀ`. No Qwen3 projection has a bias.
- "Layers" = decoder blocks; embedding, final norm and head are separate components.
- Code indices are 0-based. Human counts are written as "block i (human count i+1)" where it matters.

## Decoder block (Qwen3, dense and MoE)

```
h = x + Attn(RMSNorm_attn(x))
y = h + FFN_or_MoE(RMSNorm_ffn(h))
RMSNorm(x) = γ ⊙ x / sqrt(mean(x²) + ε)        # float32 compute, ε = 1e-6, no mean subtraction
```

Attention: `W_Q [Hq·dh, d]`, `W_K, W_V [Hkv·dh, d]`, reshape to heads, per-head RMSNorm of q and k (one `[dh]` gain each per block, shared across heads), RoPE on q and k (`rotate_half` pairing i ↔ i + dh/2, ω_i = θ^(−2i/dh), θ = 1e6), causal softmax(QKᵀ/√dh) with GQA (`repeat_kv`), concat, `W_O [d, Hq·dh]`.

SwiGLU: `Y = (SiLU(X W_gateᵀ) ⊙ X W_upᵀ) W_downᵀ`; 3·d·f parameters.

MoE (30B-A3B): `r = x W_routerᵀ`, `p = softmax(r)` over all 128 (float32), top-8, `α = p_S / Σ p_S` (`norm_topk_prob = true`), `y = Σ α_e Expert_e(x)`. No shared expert. Checkpoint tensor for the router is `mlp.gate.weight`.

## Reference configurations

| | Qwen3-8B | Qwen3-30B-A3B |
|---|---|---|
| blocks | 36 | 48 |
| d | 4096 | 2048 |
| Hq / Hkv / dh | 32 / 8 / 128 | 32 / 4 / 128 (Hq·dh = 4096 ≠ d) |
| FFN | f = 12288 | 128 experts × 768, top-8 (config `intermediate_size = 6144` is unused) |
| vocab rows | 151,936 (tokenizer: 151,669 ids) | same |
| tied | no | no |
| context | 32,768 native; 131,072 YaRN (opt-in); max_position_embeddings 40,960 | same |

## Parameter accounting (exact; equals header element totals)

| | Qwen3-8B | Qwen3-30B-A3B |
|---|---|---|
| embedding | 622,329,856 | 311,164,928 |
| head | 622,329,856 | 311,164,928 |
| attention / block | 41,943,040 | 18,874,368 |
| one expert | — | 4,718,592 |
| FFN or experts / block | 150,994,944 | 603,979,776 |
| router / block | 0 | 262,144 |
| norms / block | 8,448 | 4,352 |
| per block | 192,946,432 | 623,120,640 |
| final norm | 4,096 | 2,048 |
| **total** | **8,190,735,360** | **30,532,122,624** |

Model cards: 8.2B (6.95B non-embedding) and 30.5B (29.9B non-embedding, 3.3B activated). The "non-embedding" figures equal total − embedding − head (6,946,075,648; 29,909,792,768). Activated (MoE) conventions: selected experts over all blocks 1,811,939,328; all non-expert parameters + selected experts 3,353,032,704 (≈ card's 3.3B, but includes the whole embedding table); with one embedding row instead 3,041,869,824. None of these equals FLOPs, traffic or latency.

## KV cache

`KV = 2·B·L·T·Hkv·dh·bytes`. BF16, B = 1: 147,456 B (144 KiB)/token for 8B, 98,304 B (96 KiB)/token for 30B-A3B; at 32,768 tokens 4.5 GiB and 3 GiB. K/V only.

## Tokenizer

Byte-level BPE, 151,643 regular + 26 added ids (151643–151668: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, …, `<think>`, `</think>`). 267 embedding/head rows are padding. `enable_thinking=False` inserts `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n`.

## Qwen3.8-Flash-Next (separate profile)

- `model_type qwen4_exp`; 48 text blocks: `layer_types` = 36 `linear_attention` (Gated DeltaNet) + 12 `full_attention` (every 4th).
- The "full_attention" blocks are **QSA**: `Qwen4ExpTextQSAIndexer` mean-pools visible keys in complete 4-token blocks, scores `Σ_heads ReLU(q_h·k̄)/√128` with 4 indexer query heads + 1 key head, keeps top `2048/4 = 512` blocks plus the incomplete tail, and adds this as a mask to gated (sigmoid output gate) softmax attention. 24 Q / 2 KV heads, head dim 256, partial RoPE (64 dims).
- Gated DeltaNet recurrence (`torch_recurrent_gated_delta_rule`): `S ← S·e^{g}`, `δ = β(v − Sᵀk)`, `S ← S + kδᵀ`, `o = Sᵀq`; `g = −exp(A_log)·softplus(a + dt_bias)`, `β = σ(b)`, q/k L2-normalized, q scaled by 1/√dk; 48 V heads, 16 QK heads, dim 128, causal depthwise conv (kernel 4) before, gated RMSNorm + out_proj after.
- Residual: 4 branches × 2560 carried as `[B, T, 10240]` (`hc_count = 4`). Each sub-layer: grouped RMS norm (`hc_norm`, zero-centred gains `1 + w`), read gate `σ(W_up SiLU(W_down n / 4))` elementwise (rank 320), block input = mean of gated branches; write `branch_b += 2σ(w_b·n/4) · y`. Final `hyper_connection_mixer` (read only).
- MoE: 512 experts, 10 routed + 1 shared (width 640), shared output scaled by `σ(x·w_shared)`; `norm_topk_prob` defaults to True in `configuration_qwen4_exp.py`.
- N-gram ("PLE"): config `ple_layer_ids = [2]` is 1-based (`layer_idx + 1 in ple_layer_ids`), tensors at `model.language_model.layers.1.ple.*`. Bigram/trigram ids hashed (XOR of id × odd multipliers, mod 8 primes ≈ 20M each per order) into 16 lookups of 160 values; 128 table shards, 320,001,536 rows → 51,200,245,760 parameters.
- Header-derived components: text backbone 125,710,814,080 (card "125B"); n-gram tables + injection 51,233,085,475 (card "51B"); `mtp.*` 2,607,150,848 (card "4B MTP" — convention unstated; 3.88B if shared embedding + head are added: **Unknown**); vision 448,931,056; total stored 179,999,981,459.
- Per-sequence state (derived): QSA K/V 24,576 B/token + indexer keys 3,072 B/token (BF16), recurrent state 36 × 48 × 128 × 128 values (fixed), plus conv states. Total memory still grows with context.

## Known limitations

- No real Qwen activations, attention maps, routing decisions or recurrent states are shown. All such numbers come from the toy simulator (d = 8, 2 blocks, 2 Q / 1 KV heads, dh = 4, FFN 16 / 4 experts × 8 top-2, vocab 12, θ = 10,000) and are labelled.
- Real weight values: only seven tiles (rows 0–15 × cols 0–15 or elements 0–15) of seven tensors.
- The training toy is attention-free and uses SGD; it is not Qwen's recipe.
- Qwen3-Next and Qwen3.5 are structural views from model cards/configs; their headers were not analysed.
