import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { CHAPTERS, type Block } from '../content/chapters';
import { GLOSSARY } from '../content/glossary';
import { LAB_META } from '../labs/meta';
import { TOURS } from '../content/tours';
import { useLang, type L10n, type Lang } from '../i18n/i18n';
import { UI, type UiKey } from '../i18n/ui';

// Client-side search over chapters, labs, glossary, tours and pages, in the current language.
// Matching ignores case and diacritics ("pozornost" finds "pozornosť").

interface Doc { kind: 'chapter' | 'lab' | 'term' | 'tour' | 'page'; title: string; text: string; href: string }

const fold = (s: string) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
const strip = (s: string) => s.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*`]/g, '');

function blockText(b: Block, lang: Lang): string {
  switch (b.k) {
    case 'p': case 'h': case 'key': case 'analogy': case 'qwen': return b.t[lang];
    case 'list': return b.items.map((x) => x[lang]).join(' ');
    case 'quiz': return [b.q[lang], ...b.options.map((o) => o[lang])].join(' ');
    case 'exercise': return [b.task[lang], b.solution[lang]].join(' ');
    case 'code': return b.t?.[lang] ?? '';
    case 'table': return b.rows.flat().map((c) => c[lang]).join(' ');
    case 'math': return b.t?.[lang] ?? '';
    default: return '';
  }
}

const PAGES: [string, UiKey, L10n][] = [
  ['explore?m=qwen3-8b&lv=5', 'nav_explore', { en: 'Zoom from the whole model down to one stored weight.', sk: 'Priblížte sa od celého modelu až po jednu uloženú váhu.' }],
  ['trace', 'nav_trace', { en: 'Measured activations and attention maps of Qwen3-0.6B.', sk: 'Namerané aktivácie a mapy pozornosti Qwen3-0.6B.' }],
  ['parameters', 'nav_parameters', { en: 'Where the parameters are: treemap and exact accounting.', sk: 'Kde sú parametre: stromová mapa a presné počítanie.' }],
  ['catalog', 'nav_catalog', { en: 'Every tensor of the checkpoints with shape, dtype and byte offset.', sk: 'Každý tenzor checkpointov s tvarom, typom a bajtovým posunom.' }],
  ['pipeline', 'nav_pipeline', { en: 'Real tokenizer output and chat templates.', sk: 'Skutočný výstup tokenizéra a šablóny chatu.' }],
  ['family', 'nav_family', { en: 'The Qwen family and newer hybrid architectures.', sk: 'Rodina Qwen a novšie hybridné architektúry.' }],
  ['sources', 'nav_sources', { en: 'Sources, claims and evidence.', sk: 'Zdroje, tvrdenia a dôkazy.' }],
  ['glossary', 'nav_glossary', { en: 'Definitions of all terms.', sk: 'Definície všetkých pojmov.' }],
  ['about', 'nav_about', { en: 'About the project, limitations and your data.', sk: 'O projekte, obmedzeniach a vašich údajoch.' }],
];

function buildIndex(lang: Lang): Doc[] {
  const docs: Doc[] = [];
  for (const c of CHAPTERS) docs.push({ kind: 'chapter', title: `${c.n}. ${c.title[lang]}`, text: strip([c.summary[lang], ...c.blocks.map((b) => blockText(b, lang))].join(' ')), href: `#/learn/${c.id}` });
  for (const l of LAB_META) docs.push({ kind: 'lab', title: l.title[lang], text: l.blurb[lang], href: `#/labs/${l.id}` });
  for (const g of GLOSSARY) docs.push({ kind: 'term', title: g.term[lang], text: `${g.def[lang]} ${g.term.en}`, href: `#/glossary?q=${encodeURIComponent(g.term.en.split(' ')[0])}` });
  for (const t of TOURS) docs.push({ kind: 'tour', title: t.title[lang], text: t.steps.map((s) => s.text[lang]).join(' '), href: `#/tours/${t.id}` });
  for (const [path, key, d] of PAGES) docs.push({ kind: 'page', title: UI[key][lang], text: d[lang], href: `#/${path}` });
  return docs;
}

function snippet(text: string, words: string[]): string {
  const f = fold(text);
  let at = -1;
  for (const w of words) { const i = f.indexOf(w); if (i >= 0 && (at < 0 || i < at)) at = i; }
  if (at < 0) return text.slice(0, 160) + (text.length > 160 ? '…' : '');
  const s = Math.max(0, at - 60);
  return (s > 0 ? '…' : '') + text.slice(s, s + 180) + (s + 180 < text.length ? '…' : '');
}

function Highlight({ text, words }: { text: string; words: string[] }) {
  const f = fold(text);
  const marks: [number, number][] = [];
  for (const w of words) { let i = f.indexOf(w); while (i >= 0 && w) { marks.push([i, i + w.length]); i = f.indexOf(w, i + w.length); } }
  marks.sort((a, b) => a[0] - b[0]);
  const out: ReactNode[] = [];
  let last = 0;
  marks.forEach(([a, b], k) => { if (a < last) return; out.push(text.slice(last, a)); out.push(<mark key={k}>{text.slice(a, b)}</mark>); last = b; });
  out.push(text.slice(last));
  return <>{out}</>;
}

export default function Search({ q0 }: { q0: string }) {
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [q, setQ] = useState(q0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { input.current?.focus(); }, []);
  useEffect(() => { const h = `#/search${q ? `?q=${encodeURIComponent(q)}` : ''}`; if (window.location.hash !== h) window.history.replaceState(null, '', h); }, [q]);
  const index = useMemo(() => buildIndex(lang), [lang]);
  const words = fold(q).split(/\s+/).filter((w) => w.length >= 2);
  const results = useMemo(() => {
    if (!words.length) return [];
    return index.map((d) => {
      const ft = fold(d.title), fx = fold(d.text);
      if (!words.every((w) => ft.includes(w) || fx.includes(w))) return null;
      const score = words.reduce((s, w) => s + (ft.includes(w) ? 10 : 0) + Math.min(5, fx.split(w).length - 1), 0) + (d.kind === 'chapter' ? 1 : 0);
      return { d, score };
    }).filter((x): x is { d: Doc; score: number } => x !== null).sort((a, b) => b.score - a.score).slice(0, 40);
  }, [index, words.join(' ')]); // eslint-disable-line react-hooks/exhaustive-deps
  const KIND: Record<Doc['kind'], string> = { chapter: T('Chapter', 'Kapitola'), lab: T('Lab', 'Laboratórium'), term: T('Glossary', 'Slovník'), tour: T('Tour', 'Prehliadka'), page: T('Page', 'Stránka') };
  return (
    <div className="page">
      <h1>{T('Search', 'Hľadať')}</h1>
      <label className="field" style={{ maxWidth: 520 }}>
        {T('Search chapters, labs, glossary and pages', 'Hľadať v kapitolách, laboratóriách, slovníku a stránkach')}
        <input ref={input} type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={T('e.g. attention, KV cache, softmax', 'napr. pozornosť, cache, softmax')} />
      </label>
      {words.length > 0 && <p className="faint" role="status">{T(`${results.length} result(s)`, `Výsledky: ${results.length}`)}</p>}
      <ol style={{ listStyle: 'none', padding: 0 }}>
        {results.map(({ d }) => (
          <li key={d.href + d.title} className="card tight" style={{ marginBottom: 8 }}>
            <span className="faint mono" style={{ fontSize: '0.78rem' }}>{KIND[d.kind]}</span>
            <div><a href={d.href}><strong><Highlight text={d.title} words={words} /></strong></a></div>
            <div className="muted" style={{ fontSize: '0.9rem' }}><Highlight text={snippet(d.text, words)} words={words} /></div>
          </li>
        ))}
      </ol>
      {words.length > 0 && results.length === 0 && <p className="muted">{T('Nothing found. Try another word or the glossary.', 'Nič sa nenašlo. Skúste iné slovo alebo slovník.')}</p>}
    </div>
  );
}
