# BUILD_PROGRESS — Qwen Atlas

Brief: `prompts/claude-master-prompt.txt` (read in full, 793 lines). The supplementary bundle (START-HERE.txt, qwen-architecture-seed.json, reference-diagrams/) is **not present**; the brief is self-contained.

Execution environment: Claude Code (desktop, Windows 11), model Claude Opus 5.5 (`claude-opus-5-5`). Node 24.19.0, npm 11.17.0, Python 3.11.9, gh 2.101.0 (signed in as `sevcikro`), here.now credentials file present (`~/.herenow/credentials`, never printed or copied).

## Status

| Phase | State |
|---|---|
| A. Environment / auth inspection | done |
| B. Sources verified, revisions pinned, shape + parameter accounting with independent fixtures | done |
| C. Explorer shell, semantic levels 0–6, inspector, outline, deep links | done |
| D. Numerical engine + 12 labs | done |
| E. Catalog, sourced explanations, family map, hybrid profiles, real samples | done |
| F. Polish, mobile/keyboard, exports, error states | done |
| G. Production build + acceptance checks | done (see docs/verification.md) |
| H. GitHub push, here.now deploy, live verification | in progress — see deploy/DEPLOYMENT.md |

## Verification (latest local run)

- `npx tsc --noEmit`: 0 errors.
- `npx vitest run`: 57/57 passed (accounting incl. header cross-check, operators, toy engine, cache equivalence + invalidation, MoE invariants, sampling, finite-difference gradients, hybrid toys, sample checksums, tokenizer arithmetic).
- `npx playwright test` (desktop 1366×860 + Pixel 7, production build): 33 passed, 3 skipped by design (device-specific tests).

## Discrepancies resolved (details in docs/scientific-notes.md and the in-app Sources page)

1. Context: report 128K vs card 32,768 native + 131,072 YaRN vs config max_position_embeddings 40,960.
2. 30B-A3B `intermediate_size = 6144` is unused (all blocks MoE, width 768).
3. Per-expert checkpoint tensors vs packed runtime `gate_up_proj` / `down_proj`; counted once.
4. Model-card "non-embedding" = total − embedding − head (found during review; first draft subtracted only the embedding).
5. Qwen3.8 n-gram "layer 2" is 1-based; code index 1.
6. Qwen3.8 "full_attention" = QSA.
7. Qwen3.8 "125B" = text backbone only; "4B MTP" convention unknown (mtp.* holds 2.61B).

## Bugs found and fixed during verification

- Loss became `Infinity` for large learning rates (`-log p` after softmax underflow) → log-sum-exp.
- Absolutely-positioned screen-reader text escaped the explorer scroll container and stretched the page.
- Navigation wrapped onto four lines at 1024 px → shorter labels, menu below 1100 px.
- Toy banners lacked the literal "Illustrative simulation" label.
- Extraction script rounded 1-D sample values to 8 decimals despite claiming exact widening → fixed and re-extracted (hashes unchanged).
- Phone explorer toolbar consumed half the screen → single scrollable row.

## Remaining

- Push to GitHub, publish to here.now, run the e2e suite against the live URL, record results.

## Blockers

None so far.
