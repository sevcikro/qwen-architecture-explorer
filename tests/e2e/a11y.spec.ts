import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// Automated accessibility audit (axe-core, WCAG 2.x A/AA rules) over the main routes,
// both color themes and both languages. Serious and critical violations fail the test.
const ROUTES = [
  '/', '/#/learn', '/#/learn/attention', '/#/learn/journey', '/#/learn/forward', '/#/learn/serving', '/#/tours', '/#/explore?m=qwen3-8b&lv=5', '/#/explore?m=qwen3-8b&lv=2&b=0&t=q_proj',
  '/#/explore?m=qwen3-30b-a3b&lv=3&b=0&mod=moe', '/#/labs', '/#/labs/attention', '/#/labs/router', '/#/labs/cache', '/#/labs/memory',
  '/#/parameters', '/#/catalog', '/#/pipeline', '/#/trace', '/#/family', '/#/hybrid/qwen3.8-flash-next', '/#/glossary', '/#/sources', '/#/about',
];

for (const theme of ['light', 'dark'] as const) {
  for (const lang of ['en', 'sk'] as const) {
    test(`axe: no serious violations (${theme}, ${lang})`, async ({ page }, info) => {
      test.skip(info.project.name !== 'desktop' && theme === 'light', 'phone runs the dark theme only');
      test.setTimeout(180_000);
      await page.addInitScript(([t, l]) => {
        try { localStorage.setItem('qa-theme', t); localStorage.setItem('qa-lang', l); } catch { /* ignore */ }
      }, [theme, lang]);
      const report: string[] = [];
      for (const r of ROUTES) {
        await page.goto(r);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(200);
        const res = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
        for (const v of res.violations) {
          if (v.impact === 'serious' || v.impact === 'critical') {
            report.push(`${r} [${v.impact}] ${v.id}: ${v.help} — ${v.nodes.slice(0, 3).map((n) => `${n.target.join(' ')} {${(n.any[0]?.message ?? '').slice(0, 140)}}`).join(' | ')}`);
          }
        }
      }
      expect(report, report.join('\n')).toEqual([]);
    });
  }
}
