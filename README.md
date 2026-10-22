# Qwen Atlas

An independent, source-backed, zoomable guide to how **Qwen3-8B** (dense) and **Qwen3-30B-A3B** (mixture of experts) work, from one stored weight up to the whole model, plus verified structural views of the newer hybrid lines (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next).

> Not affiliated with or endorsed by Alibaba / the Qwen team. Model facts are cited to pinned primary sources (see `src/evidence/sources.ts` and the in-app **Sources** page).

## What is in it

- **Learn** (new in 1.1): 14 chapters for complete beginners, from one weight and one neuron to the whole model, generation, training and hardware. Each chapter has a plain explanation, an everyday comparison, a small interactive experiment (14 in total), the real Qwen3 numbers and a quick check. Progress is remembered in the browser.
- **English and Slovak** (new in 1.1): switch with the EN / SK buttons in the header, or share a link with `?lang=sk` (e.g. `#/learn/neuron?lang=sk`). The chapters, home page, navigation, tours, glossary, explorer controls, inspector headings, lab overview and evidence labels are fully translated; the detailed technical bodies of the labs, catalog and sources pages are English and say so in Slovak mode.

- **Explorer** with semantic levels 0–6: scalar → unit / gated channel → matrix tile → module → decoder block → full model → execution & learning. Viewport zoom (wheel, pinch, buttons, `+`/`-`/`0`) reveals more detail; opening a deeper level is a separate action (double-click / Enter / inspector links). Breadcrumbs, "back to containing level", outline/search tree, deep links, view export.
- **12 laboratories** sharing one seeded toy simulator: weight, unit, matrix, attention, RoPE, dense-vs-MoE, router, layer traversal, prefill/decode/cache, sampling, training, memory.
- **Parameter inventory** (treemap + exact accounting), **tensor catalog** (all 399 / 18,867 tensors, virtualized, with shard and file byte offset), **real tokenizer outputs**, **family map**, **hybrid profiles**, **4 guided tours**, **sources & claim registry**, **glossary**.
- Every number carries one of five evidence labels: Verified architecture, Derived calculation, Illustrative simulation, Actual checkpoint sample, Unknown.

## Quick start

```bash
npm ci
npm run dev            # http://localhost:5173
```

## Tests and build

```bash
npx vitest run         # 63 unit tests: accounting, operators, toy engine, cache equivalence, gradients, data provenance, translation completeness
npm run build          # writes src/generated/build-info.json from git, type-checks, builds dist/
npx playwright install chromium
npm run test:e2e       # 41 Playwright tests (desktop + Pixel 7) against `vite preview` of dist/, incl. every chapter and Slovak mode
E2E_BASE_URL=https://<live-site> npx playwright test   # same suite against a deployed site
```

`npm run build` embeds the current git commit. Build **after** committing so the identifier equals the committed source (a dirty tree is labelled "uncommitted changes" in the footer and About page).

## Data and scripts

| File | What | How produced |
|---|---|---|
| `public/data/catalog-*.json` | tensor families, shapes, dtypes, shard layout | `scripts/extract_checkpoint_metadata.py` (Range reads of safetensors headers) |
| `public/data/offsets-*.json` | per-tensor shard + byte offset | `scripts/compact_offsets.py` |
| `public/data/checkpoint-samples.json` | 7 real BF16 tiles with SHA-256 | `scripts/extract_checkpoint_metadata.py` |
| `public/data/tokenizer-examples.json` | real Qwen3 tokenizer outputs | `scripts/tokenize_examples.py` |

See `docs/checkpoint-extraction.md` for the bounded-extraction policy (100 MB budget, HTTP 206 verification, no credentials).

## Project layout

```
src/models/     verified profiles (zod-validated), shape + parameter accounting, hybrid profiles
src/math/       pure operators, toy engine, sampler, autograd training toy, memory formulas, hybrid toys
src/evidence/   source registry, claim registry, evidence classes
src/explorer/   semantic-zoom canvas, graph builders, deep views, inspector, outline
src/labs/       the 12 laboratories
src/pages/      parameters, catalog, pipeline, family, hybrid, tours, sources, glossary, about
src/content/    chapters (bilingual course), tours and glossary
src/learn/      the 14 chapter experiments
src/i18n/       language provider, interface dictionary (EN + SK, type-checked)
src/state/      hash routing + validated deep links, shared toy-simulator store
src/data/       lazy, schema-validated loaders for public/data
scripts/        extraction, tokenization, build-info
tests/unit      vitest      tests/e2e  Playwright
docs/           scientific notes, decisions, verification, extraction, deployment
```

## Deployment

Static build (`dist/`), hash routing, served from `/`. Published with the here.now helper (`publish.sh dist --spa --client claude-code`); see `docs/deployment.md`. Never commit `.herenow/` state or credentials.

## Continuing the work

Read `BUILD_PROGRESS.md` (status, verification results, blockers) and `docs/decisions.md`.

## License

Code: MIT (see `LICENSE`). The seven small weight tiles come from Apache-2.0 Qwen3 checkpoints (see `NOTICE`). Qwen3.8-Flash-Next (qwen-community-1.0 license) is described from public metadata only; none of its values are redistributed.
