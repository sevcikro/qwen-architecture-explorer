# Deployment

## here.now (primary)

1. Commit all source changes (`git status` clean).
2. `npm run build` → `dist/` with `index.html` at its root; the footer shows the commit.
3. Publish only `dist/`:

   ```bash
   ~/.claude/skills/here-now/scripts/publish.sh dist --spa --client claude-code            # first publish
   ~/.claude/skills/here-now/scripts/publish.sh dist --spa --client claude-code --slug <slug>   # updates
   ```

   The helper reads the API key from `$HERENOW_API_KEY` or `~/.herenow/credentials` (never committed, never printed) and writes private state to `.herenow/state.json` (git-ignored). Updates send the recorded `baseVersionId`; a `version_conflict` means the live site changed elsewhere — reconcile, do not `--overwrite` blindly.
4. Verify the live site: `E2E_BASE_URL=<siteUrl> npx playwright test`, plus a manual look at the footer build id.

The non-secret deployment record lives in `deploy/DEPLOYMENT.md`.

## GitHub Pages mirror (optional, not enabled)

The repository is private. A Pages mirror under `/qwen-architecture-explorer/` would need `VITE_BASE=/qwen-architecture-explorer/ npm run build` and a Pages workflow; it was not set up and the repository visibility was not changed.
