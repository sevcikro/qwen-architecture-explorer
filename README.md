# Qwen Atlas

An independent, source-backed, zoomable guide to how **Qwen3-8B** (dense) and **Qwen3-30B-A3B** (mixture of experts) work, from one stored weight up to the whole model, plus verified structural views of the newer hybrid lines (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next).

> Not affiliated with or endorsed by Alibaba / the Qwen team. Model facts are cited to pinned primary sources (see `src/evidence/sources.ts` and the in-app **Sources** page).

## What is in it

- **Learn**: an interactive mini-course of **30 chapters in five parts** (including a final test) (foundations; building blocks; the model in action; learning; practice and scale), from one stored weight to the whole transformer, the journey of a chat message, backpropagation, LLM training, serving, quantization and checkpoint files. It contains 28 explanatory diagrams, 23 interactive experiments, 16 Python examples (15 executed, with their real recorded output; also as Colab notebooks), 18 exercises with worked solutions, glossary tooltips in the text and a quick check per chapter. Progress is remembered in the browser.
- **Live model** (new in 1.3): run Qwen3-0.6B in your own browser (WebGPU or WebAssembly) after an explicit download; chat and next-token probabilities.
- **Search**, **offline use / install as an app**, share previews and a *Report a mistake* link on every page.
- **Real runs of Qwen3-0.6B** (new in 1.2): the *Real run* page shows measured activations, attention maps and gate values for three prompts; the course replays a recorded chat turn token by token (139 greedy decode steps with the top-5 candidates of each).
- **English and Slovak**: switch with the EN / SK buttons in the header, or share a link with `?lang=sk` (e.g. `#/learn/neuron?lang=sk`). Since 1.2 the whole site is translated, including labs, explorer descriptions, catalog, pipeline, family, hybrid, sources (claims and source descriptions) and about pages.
- **Accessibility**: automated axe-core audit (WCAG 2.1 A/AA rules) over 24 routes × both themes × both languages, with no serious or critical violations.

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
npx vitest run         # 89 unit tests: accounting, operators, toy engine, cache equivalence, gradients, data provenance (26 tiles re-hashed), real-run data, course structure, translation completeness
npm run build          # writes src/generated/build-info.json from git, type-checks, builds dist/
npx playwright install chromium
npm run test:e2e       # Playwright (desktop + Pixel 7) against `vite preview` of dist/: every chapter, the course extras, Slovak mode, and the axe accessibility audit
E2E_BASE_URL=https://<live-site> npx playwright test   # same suite against a deployed site
```

`npm run build` embeds the current git commit. Build **after** committing so the identifier equals the committed source (a dirty tree is labelled "uncommitted changes" in the footer and About page).

## Data and scripts

| File | What | How produced |
|---|---|---|
| `public/data/catalog-*.json` | tensor families, shapes, dtypes, shard layout | `scripts/extract_checkpoint_metadata.py` (Range reads of safetensors headers) |
| `public/data/offsets-*.json` | per-tensor shard + byte offset | `scripts/compact_offsets.py` |
| `public/data/checkpoint-samples.json` | 26 real BF16 tiles with SHA-256 | `scripts/extract_checkpoint_metadata.py` |
| `public/data/tokenizer-examples.json` | real Qwen3 tokenizer outputs | `scripts/tokenize_examples.py` |
| `public/data/trace-qwen3-0.6b.json` | measured activations of Qwen3-0.6B (3 prompts) | `scripts/record_traces.py` |
| `public/data/journey-qwen3-0.6b.json` | one recorded chat turn of Qwen3-0.6B (greedy, top-5 per step) | `scripts/record_journey.py` |
| `src/content/code-outputs.json` | recorded output of every course Python example | `scripts/run_examples.py` |

See `docs/checkpoint-extraction.md` for the bounded-extraction policy (100 MB budget, HTTP 206 verification, no credentials).

## Python examples

`examples/python/` holds the 16 programs used in the course (as notebooks in `notebooks/`, generated by `scripts/make_notebooks.py`) (NumPy, PyTorch and Hugging Face Transformers). They run on a CPU; the ones that need the model download Qwen/Qwen3-0.6B at a pinned revision, or use `QWEN_MODEL_DIR` for a local copy:

```bash
pip install numpy torch transformers
python examples/python/09_tiny_transformer.py
QWEN_MODEL_DIR=/path/to/Qwen3-0.6B python examples/python/12_hf_generate.py
```

`scripts/run_examples.py` runs them all and records their output for the site.

## Project layout

```
src/models/     verified profiles (zod-validated), shape + parameter accounting, hybrid profiles
src/math/       pure operators, toy engine, sampler, autograd training toy, memory formulas, hybrid toys
src/evidence/   source registry, claim registry, evidence classes
src/explorer/   semantic-zoom canvas, graph builders, deep views, inspector, outline
src/labs/       the 12 laboratories
src/pages/      parameters, catalog, pipeline, family, hybrid, tours, sources, glossary, about
src/content/    chapters and parts (bilingual course), Python example registry, tours and glossary
src/learn/      the 22 course experiments, the SVG diagrams and the code viewer
src/i18n/       language provider, interface dictionary (EN + SK, type-checked)
src/state/      hash routing + validated deep links, shared toy-simulator store
src/data/       lazy, schema-validated loaders for public/data
scripts/        extraction, tokenization, real-run recording, example runner, build-info
examples/python course examples
tests/unit      vitest      tests/e2e  Playwright
docs/           scientific notes, decisions, verification, extraction, deployment
```

## Deployment

Static build (`dist/`), hash routing, served from `/`. Published with the here.now helper (`publish.sh dist --spa --client claude-code`); see `docs/deployment.md`. Never commit `.herenow/` state or credentials.

## Continuing the work

Read `BUILD_PROGRESS.md` (status, verification results, blockers) and `docs/decisions.md`.

## License

Code: MIT (see `LICENSE`). The 26 small weight tiles and the recorded Qwen3-0.6B outputs come from Apache-2.0 Qwen3 models (see `NOTICE`). Qwen3.8-Flash-Next (qwen-community-1.0 license) is described from public metadata only; none of its values are redistributed.
