# Deployment record (non-secret)

| Field | Value |
|---|---|
| Host | here.now (authenticated publish, personal account) |
| Live URL | https://happy-bamboo-w7dk.here.now/ |
| Persistence | permanent (provider: `persistence=permanent`, `expiresAt: null`, status `active`) |
| Live version id | 01M5HSKMM7EH3H6PFAK7ES10T2 (update of first version 01M5HS1AS4170MV2CMMVDW15ZG) |
| Deployed source commit | 8bd039119331b0cb3e84ee3e5558713992fdb19d (embedded in the bundle, clean tree) |
| Published directory | `dist/` only (80 files), SPA routing enabled |
| Published | 2026-10-22 |

Live checks (2026-10-22): `E2E_BASE_URL=https://happy-bamboo-w7dk.here.now npx playwright test` → 33 passed, 3 skipped by design;
`/`, a deep path (SPA fallback) and `/data/*.json` return 200; `/.herenow/state.json` returns 404; the live JS bundle contains
the commit id above; no console errors, failed requests or horizontal overflow on the checked routes.

Later commits that only touch documentation (e.g. this file) do not change the deployed application source.
Private state (`.herenow/state.json`, credentials) is git-ignored and was never published.
