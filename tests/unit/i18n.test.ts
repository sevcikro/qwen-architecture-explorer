import { describe, expect, it } from 'vitest';
import { CHAPTERS, type Block } from '../../src/content/chapters';
import { GLOSSARY } from '../../src/content/glossary';
import { TOURS } from '../../src/content/tours';
import { UI } from '../../src/i18n/ui';
import type { L10n } from '../../src/i18n/i18n';

const ROUTES = ['', 'learn', 'explore', 'labs', 'parameters', 'catalog', 'pipeline', 'family', 'hybrid', 'tours', 'sources', 'glossary', 'about'];
const LABS = ['weight', 'unit', 'matrix', 'attention', 'rope', 'compare', 'router', 'layers', 'cache', 'sampling', 'training', 'memory'];
const WIDGETS = ['next-token', 'real-weight', 'neuron', 'silu', 'matvec', 'tokens', 'rms', 'attention', 'gated', 'router', 'stack', 'sampling', 'train1d', 'memory'];

function texts(b: Block): L10n[] {
  switch (b.k) {
    case 'p': case 'h': case 'key': case 'analogy': case 'qwen': return [b.t];
    case 'list': return b.items;
    case 'math': return b.t ? [b.t] : [];
    case 'quiz': return [b.q, ...b.options, b.explain];
    case 'links': return b.items.map((x) => x.label);
    default: return [];
  }
}

describe('Slovak translation completeness', () => {
  it('every UI string has non-empty English and Slovak text', () => {
    for (const [k, v] of Object.entries(UI)) {
      if (k === 'in_sk_note') { expect(v.sk.length).toBeGreaterThan(0); continue; }
      expect(v.en.trim(), k).not.toBe('');
      expect(v.sk.trim(), k).not.toBe('');
    }
  });

  it('every chapter string exists in both languages and Slovak prose is really translated', () => {
    let pairs = 0;
    for (const c of CHAPTERS) {
      for (const x of [c.title, c.summary, ...c.blocks.flatMap(texts)]) {
        expect(x.en.trim(), c.id).not.toBe('');
        expect(x.sk.trim(), c.id).not.toBe('');
        pairs++;
      }
      for (const b of c.blocks) if (b.k === 'p' || b.k === 'key' || b.k === 'qwen') expect(b.t.sk, `${c.id}: untranslated`).not.toBe(b.t.en);
      expect(c.title.sk).not.toBe(c.title.en);
    }
    expect(pairs).toBeGreaterThan(150);
  });

  it('Slovak text uses Slovak characters and decimal commas in numbers written for Slovak readers', () => {
    const all = CHAPTERS.flatMap((c) => [c.title, c.summary, ...c.blocks.flatMap(texts)]).map((x) => x.sk).join(' ');
    expect(all).toMatch(/[áäčďéíĺľňóôŕšťúýž]/);
    expect(all).toContain('8 190 735 360'); // sk-SK grouping (no-break spaces)
    expect(all).not.toMatch(/\d,\d{3},\d{3}/); // no English thousands separators leaked into Slovak
  });

  it('course structure: 14 chapters, unique ids, every chapter has a demo and a quick check, quiz answers valid', () => {
    expect(CHAPTERS.length).toBe(14);
    expect(new Set(CHAPTERS.map((c) => c.id)).size).toBe(14);
    CHAPTERS.forEach((c, i) => {
      expect(c.n).toBe(i + 1);
      expect(c.blocks.some((b) => b.k === 'widget' || c.id === 'beyond')).toBe(true);
      const quizzes = c.blocks.filter((b) => b.k === 'quiz');
      expect(quizzes.length).toBeGreaterThan(0);
      for (const q of quizzes) if (q.k === 'quiz') expect(q.correct).toBeLessThan(q.options.length);
    });
    const used = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'widget').map((b) => (b as { w: string }).w);
    for (const w of used) expect(WIDGETS).toContain(w);
    expect(new Set(used).size).toBe(WIDGETS.length); // every demo is used
  });

  it('every chapter link points to an existing route', () => {
    for (const c of CHAPTERS) for (const b of c.blocks) if (b.k === 'links') for (const l of b.items) {
      const [path] = l.href.replace(/^#\//, '').split('?');
      const [top, sub] = path.split('/');
      expect(ROUTES, l.href).toContain(top);
      if (top === 'labs' && sub) expect(LABS, l.href).toContain(sub);
      if (top === 'learn' && sub) expect(CHAPTERS.map((x) => x.id), l.href).toContain(sub);
    }
  });

  it('glossary and tours are bilingual', () => {
    for (const g of GLOSSARY) { expect(g.term.sk).not.toBe(''); expect(g.def.sk).not.toBe(g.def.en); }
    for (const t of TOURS) for (const s of t.steps) { expect(s.text.sk).not.toBe(s.text.en); expect(s.cta.sk).not.toBe(''); }
  });
});
