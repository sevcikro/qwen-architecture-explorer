import { expect, test, type Page } from '@playwright/test';
import { CHAPTERS } from '../../src/content/chapters';

// Whole-site quality sweep: every route, lab, chapter and hybrid page in both languages.
// Fails on console errors, failed requests, horizontal overflow, leaked "NaN"/"undefined"/"[object Object]"
// text, and diagram labels that leave their figure or their box.
const LABS = ['weight', 'unit', 'matrix', 'attention', 'rope', 'compare', 'router', 'layers', 'cache', 'sampling', 'training', 'memory'];
const ROUTES = [
  '/', '/#/learn', '/#/tours', '/#/labs', '/#/parameters', '/#/parameters?m=qwen3-30b-a3b', '/#/catalog', '/#/catalog?m=qwen3-30b-a3b', '/#/pipeline', '/#/trace',
  '/#/family', '/#/hybrid/qwen3-next-80b-a3b', '/#/hybrid/qwen3.5-35b-a3b', '/#/hybrid/qwen3.8-flash-next', '/#/glossary', '/#/sources', '/#/about',
  ...[5, 4, 3, 2, 1, 0, 6].map((lv) => `/#/explore?m=qwen3-8b&lv=${lv}&b=0&t=q_proj&mod=attention&r=0&c=0`),
  '/#/explore?m=qwen3-30b-a3b&lv=3&b=0&mod=moe', '/#/explore?m=qwen3-30b-a3b&lv=2&b=0&t=expert_gate_proj&e=0',
  ...LABS.map((l) => `/#/labs/${l}`),
  ...CHAPTERS.map((c) => `/#/learn/${c.id}`),
];

function watch(page: Page) {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') problems.push(`console ${m.type()}: ${m.text()}`); });
  page.on('response', (r) => { if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.url()}`); });
  return problems;
}

for (const lang of ['en', 'sk'] as const) {
  test(`QA sweep (${lang}): no errors, overflow, leaked values or clipped diagram labels`, async ({ page }) => {
    test.setTimeout(240_000);
    await page.addInitScript((l) => { try { localStorage.setItem('qa-lang', l); } catch { /* ignore */ } }, lang);
    const problems = watch(page);
    const issues: string[] = [];
    for (const r of ROUTES) {
      await page.goto(r);
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(120);
      const res = await page.evaluate(() => {
        const out: string[] = [];
        const over = document.documentElement.scrollWidth - window.innerWidth;
        if (over > 1) out.push(`horizontal overflow ${over}px`);
        const text = (document.querySelector('main')?.textContent ?? document.body.textContent ?? '').replace(/never NaN|nikdy NaN/g, '');
        for (const bad of ['NaN', 'undefined', '[object Object]', 'Infinity']) {
          const re = new RegExp(`(^|[^A-Za-z])${bad.replace(/[[\]]/g, '\\$&')}([^A-Za-z]|$)`);
          if (re.test(text)) out.push(`text contains "${bad}"`);
        }
        document.querySelectorAll('figure.fig svg').forEach((svg) => {
          const fig = svg.closest('figure')?.getAttribute('data-figure');
          const vb = (svg as SVGSVGElement).viewBox.baseVal;
          svg.querySelectorAll('text').forEach((t) => {
            if (t.getAttribute('transform')) return;
            const b = (t as SVGTextElement).getBBox();
            if (b.width === 0) return;
            if (b.x < vb.x - 1 || b.x + b.width > vb.x + vb.width + 1 || b.y < vb.y - 1 || b.y + b.height > vb.y + vb.height + 2) out.push(`figure ${fig}: label “${t.textContent}” leaves the figure`);
            const g = t.parentElement;
            const rects = g ? Array.from(g.children).filter((c) => c.tagName === 'rect') : [];
            if (g && rects.length === 1 && g.querySelectorAll(':scope > text').length <= 2) {
              const rb = (rects[0] as SVGRectElement).getBBox();
              const cx = b.x + b.width / 2;
              if (cx > rb.x && cx < rb.x + rb.width && b.y > rb.y - 2 && b.y < rb.y + rb.height && (b.x < rb.x - 1 || b.x + b.width > rb.x + rb.width + 1)) out.push(`figure ${fig}: label “${t.textContent}” overflows its box`);
            }
          });
        });
        return out;
      });
      for (const x of res) issues.push(`${r}: ${x}`);
    }
    expect([...issues, ...problems], [...issues, ...problems].join('\n')).toEqual([]);
  });
}
