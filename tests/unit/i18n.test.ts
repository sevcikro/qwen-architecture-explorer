import { describe, expect, it } from 'vitest';
import { CHAPTERS, PARTS, type Block } from '../../src/content/chapters';
import { FIGURE_IDS } from '../../src/learn/figures';
import { CODE } from '../../src/content/code';
import { GLOSSARY } from '../../src/content/glossary';
import { TOURS } from '../../src/content/tours';
import { UI } from '../../src/i18n/ui';
import type { L10n } from '../../src/i18n/i18n';

const ROUTES = ['', 'learn', 'trace', 'live', 'search', 'explore', 'labs', 'parameters', 'catalog', 'pipeline', 'family', 'hybrid', 'tours', 'sources', 'glossary', 'about'];
const LABS = ['weight', 'unit', 'matrix', 'attention', 'rope', 'compare', 'router', 'layers', 'cache', 'sampling', 'training', 'memory'];
const WIDGETS = ['next-token', 'real-weight', 'neuron', 'silu', 'matvec', 'tokens', 'rms', 'attention', 'gated', 'router', 'stack', 'sampling', 'train1d', 'memory', 'dot2d', 'xent', 'rope2d', 'gqa', 'kvcost', 'journey', 'backprop', 'quant', 'context'];

function texts(b: Block): L10n[] {
  switch (b.k) {
    case 'p': case 'h': case 'key': case 'analogy': case 'qwen': return [b.t];
    case 'figure': return b.cap ? [b.cap] : [];
    case 'code': return b.t ? [b.t] : [];
    case 'exercise': return [b.task, b.solution, ...(b.hint ? [b.hint] : [])];
    case 'table': return [...b.head, ...b.rows.flat()];
    case 'exam': return b.questions.flatMap((q) => [q.q, ...q.options, q.explain]);
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

  it('course structure: 30 chapters in 5 parts, unique ids, every chapter has a visual or demo and a quick check, quiz answers valid', () => {
    expect(CHAPTERS.length).toBe(30);
    expect(new Set(CHAPTERS.map((c) => c.id)).size).toBe(30);
    expect(PARTS.map((p) => p.n)).toEqual([1, 2, 3, 4, 5]);
    expect(PARTS.flatMap((p) => p.chapters)).toEqual(CHAPTERS.map((c) => c.id));
    expect(CHAPTERS[1].id).toBe('weight'); // the first steps keep their place
    CHAPTERS.forEach((c, i) => {
      expect(c.n).toBe(i + 1);
      expect(c.part).toBeGreaterThan(0);
      expect(c.blocks.some((b) => b.k === 'widget' || b.k === 'figure' || b.k === 'code' || b.k === 'exam' || c.id === 'beyond')).toBe(true);
      const quizzes = c.blocks.filter((b) => b.k === 'quiz' || b.k === 'exam');
      expect(quizzes.length).toBeGreaterThan(0);
      for (const q of quizzes) if (q.k === 'quiz') expect(q.correct).toBeLessThan(q.options.length);
    });
    const used = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'widget').map((b) => (b as { w: string }).w);
    for (const w of used) expect(WIDGETS).toContain(w);
    expect(new Set(used).size).toBe(WIDGETS.length); // every demo is used
    const figs = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'figure').map((b) => (b as { f: string }).f);
    for (const f of figs) expect(FIGURE_IDS).toContain(f);
    expect(new Set(figs)).toEqual(new Set(FIGURE_IDS)); // every diagram is used
    const codes = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'code').map((b) => (b as { c: string }).c);
    expect(new Set(codes)).toEqual(new Set(Object.keys(CODE))); // every Python example is used
  });

  it('every Python example has a recorded real run, except the one that needs a server', () => {
    for (const ex of Object.values(CODE)) {
      expect(ex.source.length, ex.file).toBeGreaterThan(100);
      if (ex.id === 'api-client') { expect(ex.ran).toBe(false); continue; }
      expect(ex.ran, ex.file).toBe(true);
      expect(ex.output, ex.file).not.toBe('');
      expect(ex.output, ex.file).not.toMatch(/Traceback|Error:/);
    }
    expect(CODE['tiny-transformer'].output).toContain('cache path == full recomputation: True');
    expect(CODE.backprop.output).toMatch(/by hand : dL\/dw = (-[\d.]+).*\nautograd: dL\/dw = \1/s);
    expect(CODE.safetensors.output).toContain('matches the safetensors library: True');
    expect(CODE.safetensors.output).toContain('751,632,384');
    expect(CODE['hf-inspect'].output).toContain("next token ' Paris'   p = 0.657");
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

describe('v1.3 additions', () => {
  it('every exercise and exam question is bilingual and the exam is valid', () => {
    const exams = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'exam');
    expect(exams.length).toBe(1);
    const qs = exams[0].k === 'exam' ? exams[0].questions : [];
    expect(qs.length).toBe(20);
    for (const q of qs) {
      expect(q.correct).toBeLessThan(q.options.length);
      expect(CHAPTERS.map((c) => c.id)).toContain(q.chapter);
      expect(q.q.sk).not.toBe(q.q.en);
    }
    const ex = CHAPTERS.flatMap((c) => c.blocks).filter((b) => b.k === 'exercise');
    expect(ex.length).toBeGreaterThanOrEqual(15);
    for (const e of ex) if (e.k === 'exercise') { expect(e.solution.sk).not.toBe(e.solution.en); expect(e.task.sk.length).toBeGreaterThan(10); }
  });
  it('glossary terms are found in chapter text in both languages, with sensible inflections', async () => {
    const { findTerms, mentions } = await import('../../src/content/glossary');
    expect(mentions('the K/V cache grows', 'K/V cache', 'en')).toBe(true);
    const { GLOSS_MATCH } = await import('../../src/content/glossary');
    for (const term of Object.keys(GLOSS_MATCH)) for (const l of ['en', 'sk'] as const) expect(() => mentions('x', term, l), term + ' ' + l).not.toThrow();
    expect(mentions('každá aktivácia neurónu', 'Activation', 'sk')).toBe(true);
    expect(mentions('reziduálny prúd', 'Residual connection', 'sk')).toBe(true);
    expect(mentions('tokenizér rozdelí text', 'Token', 'sk')).toBe(false);
    expect(mentions('každý z tokenov', 'Token', 'sk')).toBe(true);
    expect(mentions('pozornosť s maticou váh', 'Weight', 'sk')).toBe(true);
    const hits = findTerms('Logits become probabilities; the logits of every token.', ['Logit', 'Token'], 'en');
    expect(hits.map((h) => h.term)).toEqual(['Logit', 'Token']);
    expect(hits[0].start).toBe(0);
  });
});

