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
| H. GitHub push, here.now deploy, live verification | done — see deploy/DEPLOYMENT.md |

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

## Release 1.1.0 (learning chapters + Slovak)

- Done: 14 bilingual chapters with 14 experiments (`src/content/chapters.ts`, `src/learn/widgets.tsx`, `src/pages/Learn.tsx`), i18n system (`src/i18n/`), Slovak translation of chrome, home, tours, glossary, explorer controls, inspector headings, labs overview, evidence labels; English-only notice on detailed technical pages.
- Verified locally: tsc 0 errors; 63/63 unit tests; 41 passed + 3 skipped (by design) browser tests; screenshots reviewed (desktop dark / phone light, both languages).
- Not translated (explicitly flagged in Slovak mode): the detailed body text of the 12 labs, inspector descriptions, catalog, pipeline, family/hybrid, sources and about pages.

## Release 1.2.0 (interactive book, real runs, full Slovak, accessibility)

- Course: 27 chapters in 5 parts (`src/content/course.ts` adds 13 chapters, extra blocks for the original ones and the part order); 26 SVG diagrams (`src/learn/figures.tsx`); 8 new experiments (`src/learn/widgets2.tsx`, shared kit in `src/learn/kit.tsx`); 16 Python examples (`examples/python`) shown by `src/learn/CodeBlock.tsx` with outputs recorded by `scripts/run_examples.py` (15 executed; the API client needs a server).
- Real data: `scripts/record_traces.py` (Real run page) and `scripts/record_journey.py` (recorded chat turn, 139 greedy steps, top-5 per step), Qwen3-0.6B @ c1899de2, model file SHA-256 f47f7117… equal to the Hub LFS hash. Facts used in the text were checked against these outputs (two example labels were corrected: an overshooting learning rate, and the last hidden state being after the final norm).
- 26 checkpoint tiles (was 7), 4,120,004 bytes transferred, old tiles byte-identical.
- Slovak: every page and data text translated (`src/i18n/dataSk.ts` for claims, sources, family and hybrid texts); English-only notices removed.
- Accessibility: `tests/e2e/a11y.spec.ts` (axe-core 4.13.0, WCAG 2.1 A/AA tags, 24 routes, both themes and languages); fixes in contrast tokens, ARIA structure and keyboard-scrollable regions.
- Phone treemap fix (labels only where they fit, full segment list).

## Release 1.3.0 (live model, notebooks, search, offline, exercises, two chapters)

- Live model: `src/live/worker.ts` (Transformers.js 4.3.1 in a module worker), `src/pages/Live.tsx`, runtime files copied to `dist/ort/` by a Vite plugin (the worker's duplicate hashed copy is removed after the build). Verified in headless Chromium (WebAssembly path): model download and load, next-token probabilities, chat streaming to `<|im_end|>`, no failed requests or console errors. WebGPU path not testable here (no GPU in the test browser).
- Measured fidelity for “The capital of France is” → “ Paris”: PyTorch float32 0.657, browser q4 0.517 (still ranked first), browser q8 0.016; q4 is the WebAssembly default.
- Chapters `networks` and `context` (`src/content/course2.ts`), 18 exercises and the final test (`src/content/exercises.ts`), table block type, glossary tooltips (`src/learn/Gloss.tsx`, patterns in `src/content/glossary.ts`), search (`src/pages/Search.tsx`).
- Notebooks: `scripts/make_notebooks.py` → `notebooks/`; executed with nbclient 0.11.0 (15 of 16; the API client needs a server); the Hub download path was exercised for the tokenizer at the pinned revision.
- PWA: `public/manifest.webmanifest`, icons and `og.png` from `scripts/make_images.mjs`, service worker generated from `scripts/sw-template.js` (cache lookups ignore `Vary: Origin`, which the dev preview sends).
- Code splitting: main bundle 277 kB (was 656 kB).

## Delivery

- GitHub (public since 1.2.0): https://github.com/sevcikro/qwen-architecture-explorer, branch `main`.
- Live site (permanent): https://happy-bamboo-w7dk.here.now/ , release v1.2.0 built from commit 851ede1bf2ad (v1.1.0 was 4b8088861b9e).
- Verification for 1.2.0: tsc 0 errors; 89/89 unit tests; 53 passed + 5 skipped by design browser tests locally and against the live site (incl. site-wide QA sweep and axe audit in both languages and themes).

## Remaining / optional

- Optional: header-level parameter breakdowns for Qwen3-Next and Qwen3.5.

## Blockers

None so far.
