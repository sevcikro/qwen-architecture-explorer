# Deployment record (non-secret)

| Field | Value |
|---|---|
| Host | here.now (authenticated publish, personal account) |
| Live URL | https://happy-bamboo-w7dk.here.now/ |
| Persistence | permanent (provider: `persistence=permanent`, `expiresAt: null`, status `active`) |
| Live version id | 01M5NA1FKWZ85JHMPSMANEVWZS (release 1.3.0; 1.2.0 was 01M5JRPDPQXV5JW1EBN4NHYTZM, 1.1.0 01M5J4TWC77M9HKBAQJAC48VMD) |
| Deployed source commit | 26f3e500decdaba30df1d6ab0822ace0a7ba5e3c — release v1.3.0 (embedded in the bundle, clean tree) |
| Published directory | `dist/` only, SPA routing enabled |
| Published | 2026-10-22 |

Live checks for v1.3.0 (2026-10-23): `E2E_BASE_URL=https://happy-bamboo-w7dk.here.now npx playwright test` → 66 passed, 6 skipped by design
(desktop + Pixel 7: all 30 chapters, search, glossary tooltips, exercises and final test, live-model page without downloads,
offline mode via the service worker over https, share metadata, QA sweep and axe audit in both languages); the live bundle contains
commit 26f3e500decd; `/sw.js`, `/manifest.webmanifest`, `/og.png` and `/ort/*` return 200; `/.herenow/state.json` returns 404.
The live model was also exercised on the live site in headless Chromium (WebAssembly, q4): download from huggingface.co,
next-token probabilities, chat streamed to `<|im_end|>`, no failed requests or console errors.

Live checks for v1.2.0 (2026-10-22): `E2E_BASE_URL=https://happy-bamboo-w7dk.here.now npx playwright test` → 53 passed, 5 skipped by design
(desktop + Pixel 7: all 27 chapters, course extras, Slovak mode, the site-wide QA sweep in both languages and the axe accessibility audit);
the live bundle contains commit 851ede1bf2ad; `/.herenow/state.json` returns 404; an SPA deep path and `/data/journey-qwen3-0.6b.json` return 200.

Live checks for v1.1.0 (2026-10-22): `E2E_BASE_URL=https://happy-bamboo-w7dk.here.now npx playwright test` → 41 passed, 3 skipped by design (incl. all 14 chapters and Slovak mode);
`/`, a deep path (SPA fallback) and `/data/*.json` return 200; `/.herenow/state.json` returns 404; the live JS bundle contains
the commit id above; no console errors, failed requests or horizontal overflow on the checked routes.

Later commits that only touch documentation (e.g. this file) do not change the deployed application source.
Private state (`.herenow/state.json`, credentials) is git-ignored and was never published.
