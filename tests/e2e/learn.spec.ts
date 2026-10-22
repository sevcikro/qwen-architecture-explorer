import { expect, test, type Page } from '@playwright/test';
import { CHAPTERS } from '../../src/content/chapters';

function watch(page: Page) {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
  page.on('response', (r) => { if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.url()}`); });
  return problems;
}

test('course: start, answer a quick check, move to the next chapter', async ({ page }) => {
  const problems = watch(page);
  await page.goto('/');
  await page.getByTestId('start-learn').click();
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Learn: from one weight');
  await page.getByTestId('learn-start').click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('What a language model actually does');
  await page.getByRole('radio', { name: /No, the weights stay fixed/ }).click();
  await expect(page.getByRole('status')).toContainText('Correct.');
  await page.getByRole('link', { name: /Next → 2\./ }).click();
  await expect(page).toHaveURL(/#\/learn\/weight/);
  await expect(page.getByText('0.000568389892578125').first()).toBeVisible();
  expect(problems).toEqual([]);
});

test('every chapter renders its text, demo and quick check without errors', async ({ page }) => {
  const problems = watch(page);
  for (const c of CHAPTERS) {
    await page.goto(`/#/learn/${c.id}`);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(c.title.en);
    await expect(page.getByText('Quick check').first()).toBeVisible();
    if (c.blocks.some((b) => b.k === 'widget')) await expect(page.locator('figure.widget').first()).toBeVisible();
    const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(over, `overflow in ${c.id}`).toBeLessThanOrEqual(1);
  }
  expect(problems).toEqual([]);
});

test('demos compute: neuron output follows its weights; one training step lowers the loss', async ({ page }) => {
  await page.goto('/#/learn/neuron');
  const out = page.locator('figure.widget').first().locator('p.mono strong');
  const before = await out.textContent();
  await page.getByRole('button', { name: 'no activation' }).click();
  await expect(out).not.toHaveText(before ?? '');
  await page.goto('/#/learn/learning');
  const lossText = () => page.locator('figure.widget p.mono').first().textContent();
  const l0 = Number((await lossText())!.match(/loss = ([\d.]+)/)![1]);
  await page.getByRole('button', { name: 'One training step' }).click();
  const l1 = Number((await lossText())!.match(/loss = ([\d.]+)/)![1]);
  expect(l1).toBeLessThan(l0);
});

test('Slovak: switch language, everything core is translated, choice persists, deep link works', async ({ page, browser }) => {
  const problems = watch(page);
  await page.goto('/#/learn/what');
  await page.getByRole('button', { name: 'Slovenčina', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'sk');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Čo jazykový model v skutočnosti robí');
  await expect(page.getByText('Rýchla kontrola').first()).toBeVisible();
  await page.getByRole('radio', { name: /Nie, váhy zostávajú rovnaké/ }).click();
  await expect(page.getByRole('status')).toContainText('Správne.');
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Čo jazykový model v skutočnosti robí');
  await page.goto('/#/explore?m=qwen3-8b&lv=4&b=0');
  await expect(page.getByRole('button', { name: /Zdieľať pohľad/ })).toBeVisible();
  await page.goto('/#/glossary');
  await expect(page.getByRole('heading', { name: 'Slovník pojmov' })).toBeVisible();
  await page.goto('/#/labs/attention');
  await expect(page.getByText(/Ilustračná simulácia\./).first()).toBeVisible();
  await page.getByRole('button', { name: 'English', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  // a fresh visitor following a Slovak link
  const ctx = await browser.newContext();
  const p2 = await ctx.newPage();
  await p2.goto('/#/learn/neuron?lang=sk');
  await expect(p2.getByRole('heading', { level: 1 })).toHaveText('Jeden neurón: vážený súčet');
  await ctx.close();
  expect(problems).toEqual([]);
});

test('course extras: diagrams, Python code with its recorded output, and the real message journey', async ({ page }) => {
  const problems = watch(page);
  await page.goto('/#/learn');
  await expect(page.getByRole('heading', { level: 2, name: /Part 3 · The model in action/ })).toBeVisible();
  await page.goto('/#/learn/attention');
  await expect(page.locator('figure.fig[data-figure="qkv"] svg')).toBeVisible();
  await expect(page.locator('figure.code-card pre.out').first()).toContainText('row sums: [1. 1. 1. 1.]');
  await page.goto('/#/learn/journey');
  const w = page.locator('figure.widget').first();
  await expect(w).toContainText('Real sample');
  await w.getByRole('button', { name: /7 · Decode loop/ }).click();
  await expect(w).toContainText('Answer token 1 of 139');
  await w.getByRole('button', { name: /9 · Stop/ }).click();
  await expect(w).toContainText('<|im_end|> after 139 tokens');
  await expect(w).toContainText('A **parameter** is a value');
  await page.goto('/#/learn/hf?lang=sk');
  await page.reload();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Spustenie a preskúmanie skutočného modelu Qwen');
  await expect(page.locator('figure.code-card pre.out').first()).toContainText("next token ' Paris'   p = 0.657");
  expect(problems).toEqual([]);
});
