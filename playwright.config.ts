import { defineConfig, devices } from '@playwright/test';

// Runs against the production build (vite preview) by default; set E2E_BASE_URL to test a deployed site.
const external = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 45_000,
  expect: { timeout: 8_000 },
  fullyParallel: true,
  reporter: [['list']],
  use: { baseURL: external ?? 'http://localhost:4173', trace: 'off' },
  webServer: external ? undefined : { command: 'npx vite preview --port 4173 --strictPort', port: 4173, reuseExistingServer: true, timeout: 60_000 },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1366, height: 860 } } },
    { name: 'phone', use: { ...devices['Pixel 7'] } },
  ],
});
