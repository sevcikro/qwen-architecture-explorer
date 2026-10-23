import { expect, test } from '@playwright/test';

test('search finds chapters in both languages, ignoring diacritics', async ({ page }) => {
  await page.goto('/#/search?q=context%20window');
  await expect(page.getByRole('searchbox')).toHaveValue('context window');
  await expect(page.getByRole('link', { name: /The context window: input and output limits/ })).toBeVisible();
  await page.getByRole('button', { name: 'Slovenčina', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'sk');
  await page.goto('/#/search');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Hľadať');
  await page.getByRole('searchbox').fill('kontextove okno');
  await expect(page.getByRole('link', { name: /Kontextové okno: limity vstupu a výstupu/ })).toBeVisible();
  await expect(page.getByRole('status')).toContainText('Výsledky');
});

test('the header has a search button on wide screens', async ({ page }, info) => {
  test.skip(info.project.name !== 'desktop', 'icon is shown on desktop; phones use the menu');
  await page.goto('/');
  await page.getByRole('link', { name: 'Search' }).first().click();
  await expect(page).toHaveURL(/#\/search/);
  await expect(page.getByRole('heading', { name: 'Search', level: 1 })).toBeVisible();
});

test('glossary terms in chapters open a definition', async ({ page }) => {
  await page.goto('/#/learn/attention');
  const term = page.locator('.gloss-term').first();
  await term.click();
  await expect(page.getByRole('tooltip')).toBeVisible();
  await expect(page.getByRole('tooltip').getByRole('link')).toHaveAttribute('href', /#\/glossary\?q=/);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
});

test('exercises reveal worked solutions; the final test scores answers', async ({ page }) => {
  await page.goto('/#/learn/neuron');
  const ex = page.locator('.exercise').first();
  await ex.getByText('Show the worked solution').click();
  await expect(ex).toContainText('−0.189');
  await page.goto('/#/learn/exam');
  await page.locator('fieldset').nth(0).getByLabel('A probability for every entry of its vocabulary').check();
  await page.locator('fieldset').nth(1).getByLabel('2', { exact: true }).check();
  await page.getByRole('button', { name: /Check my answers \(2 of 20 answered\)/ }).click();
  await expect(page.getByRole('status')).toContainText('Score: 2 / 20');
});

test('live model page explains the download and fetches nothing from huggingface.co before the click', async ({ page }) => {
  const external: string[] = [];
  page.on('request', (r) => { if (!r.url().startsWith('http://localhost') && !r.url().startsWith(process.env.E2E_BASE_URL ?? 'http://localhost')) external.push(r.url()); });
  await page.goto('/#/live');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Run a real Qwen model in your browser');
  await expect(page.getByTestId('live-start')).toBeVisible();
  await expect(page.getByText(/directly from huggingface.co/)).toBeVisible();
  await page.waitForTimeout(500);
  expect(external).toEqual([]);
});

test('works offline after the first visit', async ({ page, context }) => {
  test.skip(!!process.env.E2E_BASE_URL && !process.env.E2E_BASE_URL.startsWith('https://'), 'service workers need https or localhost');
  await page.goto('/#/about');
  await expect(page.getByTestId('offline-status')).toContainText(/Offline copy active/, { timeout: 20_000 });
  await context.setOffline(true);
  await page.goto('/#/learn/context');
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('The context window: input and output limits');
  await expect(page.locator('figure.fig[data-figure="context"] svg')).toBeVisible();
  await context.setOffline(false);
});

test('share preview, app manifest and report link', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', 'https://happy-bamboo-w7dk.here.now/og.png');
  const m = await request.get('/manifest.webmanifest');
  expect(m.ok()).toBe(true);
  expect((await m.json()).name).toBe('Qwen Atlas');
  expect((await request.get('/og.png')).ok()).toBe(true);
  const report = page.getByRole('link', { name: 'Report a mistake' });
  await expect(report).toHaveAttribute('href', /github\.com\/sevcikro\/qwen-architecture-explorer\/issues\/new\?title=/);
});
