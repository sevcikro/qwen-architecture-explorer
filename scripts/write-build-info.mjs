// Writes src/generated/build-info.json from git so the compiled site carries the
// exact source commit it was built from. Run AFTER committing; a dirty tree is
// recorded as dirty=true so it can never masquerade as a clean commit build.
import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';

const sh = (c) => { try { return execSync(c, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return ''; } };
const commit = sh('git rev-parse HEAD') || 'uncommitted';
const status = sh('git status --porcelain --untracked-files=no');
const info = {
  commit,
  shortCommit: commit.slice(0, 12),
  dirty: status.length > 0,
  builtAt: new Date().toISOString(),
};
mkdirSync('src/generated', { recursive: true });
writeFileSync('src/generated/build-info.json', JSON.stringify(info, null, 2) + '\n');
console.log(`build-info: ${info.shortCommit}${info.dirty ? ' (dirty)' : ''}`);
