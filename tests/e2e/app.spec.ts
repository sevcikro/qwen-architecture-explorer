import { expect, test, type Page } from '@playwright/test';

// Collect uncaught errors, console errors and failed requests for every test.
function watch(page: Page) {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
  page.on('requestfailed', (r) => problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
  page.on('response', (r) => { if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.url()}`); });
  return problems;
}

const isPhone = (name: string) => name === 'phone';

test('home renders, start-with-one-weight reaches a real checkpoint value', async ({ page }) => {
  const problems = watch(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Where are the weights');
  await page.getByTestId('start-weight').click();
  await expect(page).toHaveURL(/lv=0/);
  await expect(page.getByText('0.000568389892578125').first()).toBeVisible();
  await expect(page.getByText('Real sample').first()).toBeVisible();
  await expect(page.getByText(/byte 1,588,618,872/)).toBeVisible();
  expect(problems).toEqual([]);
});

test('top-down drill-down from the full model to one scalar, with breadcrumbs', async ({ page }, info) => {
  const problems = watch(page);
  await page.goto('/#/explore?m=qwen3-8b&lv=5');
  await page.locator('[data-node="block:3"]').dblclick();
  await expect(page).toHaveURL(/lv=4&b=3/);
  await page.locator('[data-node="attention"]').dblclick();
  await expect(page).toHaveURL(/lv=3&b=3&mod=attention/);
  await page.locator('[data-node="tensor:k_proj"]').dblclick();
  await expect(page).toHaveURL(/lv=2&b=3&t=k_proj/);
  await expect(page.locator('.matrix-grid .cell.na')).toHaveCount(256); // real values not loaded here, never invented
  await page.getByRole('gridcell', { name: 'row 2, column 5: value not loaded' }).click();
  await expect(page).toHaveURL(/lv=0.*r=2&c=5/);
  await expect(page.getByText('Actual value not loaded.')).toBeVisible();
  const crumbs = page.getByRole('navigation', { name: 'Breadcrumb' });
  await expect(crumbs).toContainText('Block 3');
  await expect(crumbs).toContainText('k_proj');
  await crumbs.getByRole('button', { name: 'Block 3' }).click();
  await expect(page).toHaveURL(/lv=4&b=3$/);
  if (!isPhone(info.project.name)) await expect(page.getByRole('complementary', { name: 'Inspector' })).toContainText('Decoder block 3');
  expect(problems).toEqual([]);
});

test('keyboard-only: Tab to a node, Enter opens a deeper semantic level', async ({ page }, info) => {
  test.skip(isPhone(info.project.name), 'keyboard flow checked on desktop');
  await page.goto('/#/explore?m=qwen3-30b-a3b&lv=4&b=0');
  const moe = page.locator('[data-node="moe"]');
  await moe.focus();
  await expect(moe).toBeFocused();
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/lv=3&b=0&mod=moe/);
  await page.locator('[data-node="expert:17"]').focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/lv=2&b=0&t=expert_gate_proj&e=17/);
  await page.getByText('Logical expert matrices ↔ physical tensors').click();
  await expect(page.getByText('gate_up_proj[17, 0:768, :]')).toBeVisible();
});

test('model switching translates the selection and never shows stale values', async ({ page }) => {
  await page.goto('/#/explore?m=qwen3-8b&lv=2&b=35&t=gate_proj');
  await page.getByLabel('Model').selectOption('qwen3-30b-a3b');
  await expect(page).toHaveURL(/m=qwen3-30b-a3b/);
  await expect(page.getByRole('status')).toContainText('does not exist in Qwen3-30B-A3B');
  await expect(page).not.toHaveURL(/t=gate_proj/);
  await page.getByLabel('Model').selectOption('qwen3-8b');
  await expect(page.getByText('Qwen3-8B', { exact: true }).first()).toBeVisible();
});

test('deep links survive reload; invalid links are corrected with an explanation', async ({ page }) => {
  await page.goto('/#/explore?m=qwen3-30b-a3b&lv=1&b=12&t=o_proj&r=7');
  await expect(page.getByRole('heading', { name: /Output coordinate 7/ })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: /Output coordinate 7/ })).toBeVisible();
  await page.goto('/#/explore?m=qwen3-8b&lv=4&b=99');
  await expect(page.getByRole('status')).toContainText('Block 99 does not exist');
  await expect(page).toHaveURL(/lv=5/);
});

test('semantic zoom reveals more detail than viewport scaling', async ({ page }, info) => {
  test.skip(isPhone(info.project.name), 'zoom buttons checked on desktop');
  await page.goto('/#/explore?m=qwen3-8b&lv=5');
  const subs = () => page.locator('[data-node="block:0"] .node-sub').count();
  const before = await subs();
  for (let i = 0; i < 4; i++) await page.getByRole('button', { name: 'Zoom in' }).click();
  expect(await subs()).toBeGreaterThan(before);
  await expect(page.locator('[data-node="block:0"]')).toContainText('model.layers.0.*');
});

test('attention lab: causal mask and normalized rows', async ({ page }) => {
  const problems = watch(page);
  await page.goto('/#/labs/attention');
  await expect(page.getByText('Illustrative simulation').first()).toBeVisible();
  await page.getByLabel('query position').fill('2');
  const rows = page.locator('table.t').first().locator('tbody tr');
  await expect(rows.filter({ hasText: 'future' })).toHaveCount(3);
  await expect(page.locator('table.t').first()).toContainText('1.0000');
  expect(problems).toEqual([]);
});

test('cache lab: cached decoding matches full recomputation; invalidation after a prefix edit', async ({ page }) => {
  await page.goto('/#/labs/cache');
  await expect(page.getByText('✓ All decode steps match.')).toBeVisible();
  await page.getByRole('button', { name: 'Synchronize cache with current prompt' }).click();
  await expect(page.getByText(/Decoded 6 new position/)).toBeVisible();
  await page.getByLabel('token at position 2').selectOption({ label: 'dog' });
  await page.getByRole('button', { name: 'Synchronize cache with current prompt' }).click();
  await expect(page.getByText(/kept 2 cached positions, dropped 4/)).toBeVisible();
  await expect(page.getByText(/✓ last-position logits from the cache path vs full recomputation/)).toBeVisible();
});

test('router lab: official-size fixture selects 8 of 128 with coefficients summing to 1', async ({ page }) => {
  await page.goto('/#/labs/router');
  await page.getByRole('button', { name: 'Switch to the MoE toy' }).click();
  await expect(page.getByText(/router logits r = x W_router/)).toBeVisible();
  await expect(page.locator('.expert-grid button.sel')).toHaveCount(8);
  await expect(page.getByText(/renormalized Σ α = 1\.0000/)).toBeVisible();
});

test('weight lab: edits are explicit, flagged and resettable', async ({ page }) => {
  await page.goto('/#/labs/weight');
  await page.getByRole('button', { name: 'flip sign' }).click();
  await expect(page.getByText(/Downstream effect of this one change/)).toBeVisible();
  await expect(page.getByText('block 0 output: max change 0 (unaffected)')).toBeVisible();
  await page.getByRole('button', { name: /Apply this edit/ }).click();
  await expect(page.getByText(/Edited weights \(1\)/)).toBeVisible();
  await page.getByRole('button', { name: 'restore fixture' }).click();
  await expect(page.getByText(/Edited weights/)).toHaveCount(0);
});

test('sampling lab: greedy mode and seeded draws that only advance on demand', async ({ page }) => {
  await page.goto('/#/labs/sampling');
  await page.getByRole('button', { name: 'Greedy (T = 0)' }).click();
  await expect(page.getByText('(greedy: argmax gets probability 1)')).toBeVisible();
  await expect(page.getByText(/draws so far: 0/)).toBeVisible();
  await page.getByRole('button', { name: 'Draw next token (seeded)' }).click();
  await expect(page.getByText(/draws so far: 1/)).toBeVisible();
});

test('training lab: gradient passes finite differences; a step recomputes the loss', async ({ page }) => {
  await page.goto('/#/labs/training');
  await expect(page.getByText(/relative difference = .* ✓/)).toBeVisible();
  await page.getByRole('button', { name: /Take one step/ }).click();
  await expect(page.locator('table.t').filter({ hasText: 'loss after (recomputed)' })).toContainText('decreased');
});

test('memory lab reproduces the KV formulas', async ({ page }) => {
  await page.goto('/#/labs/memory');
  await expect(page.getByText('= 4.832 GB = 4.500 GiB')).toBeVisible();
  await page.getByLabel('Model').selectOption('qwen3-30b-a3b');
  await expect(page.getByText('= 3.221 GB = 3.000 GiB')).toBeVisible();
});

test('parameter inventory totals', async ({ page }) => {
  await page.goto('/#/parameters');
  await expect(page.getByText('8,190,735,360').first()).toBeVisible();
  await page.getByLabel('Model').selectOption('qwen3-30b-a3b');
  await expect(page.getByText('30,532,122,624').first()).toBeVisible();
  await expect(page.getByText('3,353,032,704').first()).toBeVisible();
});

test('catalog is virtualized and lists all 18,867 tensors of Qwen3-30B-A3B', async ({ page }) => {
  await page.goto('/#/catalog?m=qwen3-30b-a3b');
  const table = page.getByRole('table', { name: 'Tensor catalog (virtualized)' });
  await expect(table).toHaveAttribute('aria-rowcount', '18867');
  expect(await table.getByRole('row').count()).toBeLessThan(40);
  await expect(page.getByText('18867 tensors in 16 shards', { exact: false })).toBeVisible();
});

test('hybrid page and pipeline render with sources', async ({ page }) => {
  const problems = watch(page);
  await page.goto('/#/hybrid/qwen3.8-flash-next');
  await expect(page.getByRole('heading', { name: 'Qwen3.8-Flash-Next', level: 1 })).toBeVisible();
  await expect(page.getByText('125,710,814,080').first()).toBeVisible();
  await page.goto('/#/pipeline');
  await expect(page.getByText('151667').first()).toBeVisible();
  await page.goto('/#/sources');
  await expect(page.getByRole('heading', { name: 'Resolved discrepancies' })).toBeVisible();
  expect(problems).toEqual([]);
});

test('no horizontal page scroll and the build id is shown', async ({ page }) => {
  for (const path of ['/', '/#/labs', '/#/labs/attention', '/#/labs/memory', '/#/parameters', '/#/catalog', '/#/pipeline', '/#/hybrid/qwen3.8-flash-next', '/#/sources', '/#/explore?m=qwen3-8b&lv=2&b=0&t=q_proj']) {
    await page.goto(path);
    await page.waitForTimeout(150);
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(over, `horizontal overflow on ${path}`).toBeLessThanOrEqual(1);
  }
  await page.goto('/#/about');
  await expect(page.getByTestId('build-id')).toContainText('build ');
});

test('phone: menu and bottom-sheet inspector are reachable', async ({ page }, info) => {
  test.skip(!isPhone(info.project.name), 'phone only');
  await page.goto('/');
  await page.getByRole('button', { name: '☰ Menu' }).click();
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Explorer' }).click();
  await expect(page).toHaveURL(/#\/explore/);
  const toggle = page.getByRole('button', { name: /Details:/ });
  await toggle.click();
  await expect(page.getByRole('button', { name: 'Hide details ▾' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'What is this?' })).toBeInViewport();
});
