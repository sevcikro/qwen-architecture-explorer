# Deployment record (non-secret)

| Field | Value |
|---|---|
| Host | here.now (authenticated publish, personal account) |
| Live URL | https://happy-bamboo-w7dk.here.now/ |
| Persistence | permanent (provider: `persistence=permanent`, `expiresAt: null`, status `active`) |
| Live version id | 01M5J4TWC77M9HKBAQJAC48VMD (release 1.1.0; earlier versions 01M5HSKMM7EH3H6PFAK7ES10T2, 01M5HS1AS4170MV2CMMVDW15ZG) |
| Deployed source commit | 4b8088861b9e0bba56e8f1483b1e0c81bebb4f2e — release v1.1.0 (embedded in the bundle, clean tree) |
| Published directory | `dist/` only (80 files), SPA routing enabled |
| Published | 2026-10-22 |

Live checks for v1.1.0 (2026-10-22): `E2E_BASE_URL=https://happy-bamboo-w7dk.here.now npx playwright test` → 41 passed, 3 skipped by design (incl. all 14 chapters and Slovak mode);
`/`, a deep path (SPA fallback) and `/data/*.json` return 200; `/.herenow/state.json` returns 404; the live JS bundle contains
the commit id above; no console errors, failed requests or horizontal overflow on the checked routes.

Later commits that only touch documentation (e.g. this file) do not change the deployed application source.
Private state (`.herenow/state.json`, credentials) is git-ignored and was never published.
