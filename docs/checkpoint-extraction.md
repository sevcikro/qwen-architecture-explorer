# Bounded checkpoint metadata and sample extraction

Script: `scripts/extract_checkpoint_metadata.py` (Python standard library only; run with `python -I`).

```bash
python -I scripts/extract_checkpoint_metadata.py --out /tmp/qa-extract            # headers of all 3 models + the tiles in SAMPLE_TILES
python -I scripts/extract_checkpoint_metadata.py --out /tmp/qa-extract --no-samples --models qwen3-8b
python -I scripts/compact_offsets.py /tmp/qa-extract public/data qwen3-8b qwen3-30b-a3b
```

## Policy

- Pinned revisions only (see `MODELS` in the script). Public repositories; **no token is read or sent**.
- Aggregate budget (default **100 MB**, `--budget-mb`) is reserved *before* any response body is accepted.
- Every range response must be HTTP **206** with the exact requested `Content-Range` and `Content-Length`; otherwise that path stops (a server ignoring Range can never trigger a full-file download).
- 45 s timeout per request; up to 4 attempts on transient network errors, never on protocol violations.
- Only the 8-byte length prefix and the JSON header of each shard are read, plus 16 row-slices of 32 bytes for each tile.

## Result of the run used for this site (2026-10-22)

- Bytes transferred: 4,512,682 (index files + 152 shard headers + 7 tiles).
- Qwen3-8B: 399 tensors / 5 shards / 8,190,735,360 elements (= index total_size / 2).
- Qwen3-30B-A3B: 18,867 tensors / 16 shards / 30,532,122,624 elements.
- Qwen3.8-Flash-Next: 1,658 tensors / 131 shards / 179,999,981,459 elements (metadata only).
- Tiles: q_proj, gate_proj, embed_tokens, input_layernorm of 8B block 0; q_proj, router (mlp.gate), expert 0 gate_proj of 30B block 0.
- A re-run produced byte-identical catalogs and identical tile hashes.

## v1.2 run (2026-10-22): more tiles

- `SAMPLE_TILES` extended to 26 tiles; run with `--models qwen3-8b,qwen3-30b-a3b`.
- Bytes transferred: 4,120,004 (2 index files + 21 shard headers + 26 tiles) of the 100 MB budget.
- The 7 tiles of v1.0 came back with identical tensors, values and SHA-256 hashes.
- New tiles: Qwen3-8B block 0 k_proj, v_proj, o_proj, q_norm, k_norm, up_proj, down_proj, post_attention_layernorm; q_proj of blocks 17 and 35; model.norm; lm_head. Qwen3-30B-A3B block 0 expert 0 up_proj and down_proj, experts 1 and 127 gate_proj, k_proj, embed_tokens; the router of block 47.
- Only `samples.json` was copied into `public/data/checkpoint-samples.json`; catalogs and offsets were left unchanged.

## Sample record

Each tile stores model, revision, shard, tensor, logical shape, index ranges, dtype, conversion ("BF16 widened exactly to float32"), values, SHA-256 of the raw bytes, byte-range method, retrieval date, license and a note that it is not a model or a trace. `tests/unit/data.test.ts` rebuilds the raw bytes from the values and re-checks every hash, shard and shape.

The 2-byte value at a computed file offset was also cross-checked by hand: Qwen3-8B `model.layers.0.self_attn.q_proj.weight[0, 0]` at byte 1,588,618,872 of `model-00001-of-00005.safetensors` reads `15 3a` = 0x3a15 = 0.000568389892578125.

## Real tokenizer outputs

`scripts/tokenize_examples.py` (requires `tokenizers==0.22.1` and `jinja2` in an isolated virtual environment) renders the pinned chat template and tokenizes five fixed prompts. Output records the tokenizer.json SHA-256 and library version. No model is run.
