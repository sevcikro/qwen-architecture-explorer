import { Fragment, createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { UI, type UiKey } from './ui';
import { GB, GiB, fmtBytes } from '../math/memory';

export type Lang = 'en' | 'sk';
export const LANGS: Lang[] = ['en', 'sk'];
/** Every user-facing string that is translated carries both languages; the type makes a missing one a compile error. */
export interface L10n { en: string; sk: string }

function initialLang(): Lang {
  try {
    const q = new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('lang');
    if (q === 'sk' || q === 'en') return q;
    const saved = localStorage.getItem('qa-lang');
    if (saved === 'sk' || saved === 'en') return saved;
  } catch { /* storage may be unavailable */ }
  const nav = (typeof navigator !== 'undefined' ? navigator.language : 'en').toLowerCase();
  return nav.startsWith('sk') || nav.startsWith('cs') ? 'sk' : 'en';
}

interface LangApi {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (k: UiKey) => string;
  tr: (x: L10n) => string;
  /** locale-aware integer / decimal formatting (Slovak uses spaces and a decimal comma) */
  num: (n: number, digits?: number) => string;
  /** compact scientific-style number (like fmtNum) with the locale's decimal separator */
  fnum: (x: number, digits?: number) => string;
  /** shape [a, b] with locale grouping */
  shape: (s: number[]) => string;
  /** byte size like fmtBytes, localized */
  bytes: (b: number) => string;
  /** compact parameter count like fmtB (8.191B), localized (8,191 mld.) */
  big: (n: number) => string;
}

export function fmtNumL(lang: Lang, x: number, digits = 4): string {
  let out: string;
  if (!Number.isFinite(x)) out = String(x);
  else if (x === 0) out = '0';
  else if (Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3) out = x.toExponential(digits - 1);
  else out = x.toFixed(digits);
  return lang === 'sk' ? out.replace('.', ',') : out;
}

export function fmtBytesL(lang: Lang, b: number): string {
  if (lang !== 'sk') return fmtBytes(b);
  const gb = b / GB, gib = b / GiB;
  const c = (x: string) => x.replace('.', ',');
  if (b < 1e6) return `${c((b / 1024).toFixed(b < 1e4 ? 2 : 1))} KiB (${formatNum('sk', b)} bajtov)`;
  return `${c(gb.toFixed(gb < 10 ? 3 : 2))} GB = ${c(gib.toFixed(gib < 10 ? 3 : 2))} GiB`;
}

/** Same thresholds as fmtB in components/ui. */
export function fmtBL(lang: Lang, n: number): string {
  const sk = lang === 'sk';
  const c = (x: string) => (sk ? x.replace('.', ',') : x);
  return n >= 1e9 ? `${c((n / 1e9).toFixed(3))}${sk ? ' mld.' : 'B'}` : n >= 1e6 ? `${c((n / 1e6).toFixed(2))}${sk ? ' mil.' : 'M'}` : n >= 1e3 ? `${c((n / 1e3).toFixed(1))}${sk ? ' tis.' : 'K'}` : String(n);
}

const Ctx = createContext<LangApi | null>(null);

export const formatNum = (lang: Lang, n: number, digits?: number) =>
  n.toLocaleString(lang === 'sk' ? 'sk-SK' : 'en-US', digits === undefined ? undefined : { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try { localStorage.setItem('qa-lang', l); } catch { /* ignore */ }
  }, []);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  const api = useMemo<LangApi>(() => ({
    lang,
    setLang,
    t: (k) => UI[k][lang],
    tr: (x) => x[lang],
    num: (n, d) => formatNum(lang, n, d),
    fnum: (x, d) => fmtNumL(lang, x, d),
    shape: (sh) => `[${sh.map((v) => formatNum(lang, v)).join(', ')}]`,
    bytes: (b) => fmtBytesL(lang, b),
    big: (n) => fmtBL(lang, n),
  }), [lang, setLang]);
  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useLang(): LangApi {
  const c = useContext(Ctx);
  if (!c) throw new Error('LangProvider missing');
  return c;
}

/**
 * Minimal inline markup for translated prose: **bold**, *italic*, `code`, [label](href).
 * No HTML is ever injected.
 */
export interface GlossOptions { terms: string[]; find: (text: string, terms: string[]) => { term: string; start: number; end: number }[]; render: (term: string, text: string, key: string) => ReactNode }

/** Plain text with the first mention of each wanted glossary term turned into a tooltip trigger. */
function glossed(text: string, g: GlossOptions | undefined, left: Set<string>, key: string): ReactNode {
  if (!g || left.size === 0) return text;
  const hits = g.find(text, [...left]);
  if (!hits.length) return text;
  const out: ReactNode[] = [];
  let last = 0;
  hits.forEach((h, i) => {
    if (h.start > last) out.push(text.slice(last, h.start));
    out.push(g.render(h.term, text.slice(h.start, h.end), `${key}-g${i}`));
    left.delete(h.term);
    last = h.end;
  });
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Rich({ text, gloss }: { text: string; gloss?: GlossOptions }) {
  const left = new Set(gloss?.terms ?? []);
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={`t${i++}`}>{glossed(text.slice(last, m.index), gloss, left, `t${i}`)}</Fragment>);
    const s = m[0];
    if (s.startsWith('**')) out.push(<strong key={i++}>{glossed(s.slice(2, -2), gloss, left, `b${i}`)}</strong>);
    else if (s.startsWith('`')) out.push(<code key={i++}>{s.slice(1, -1)}</code>);
    else if (s.startsWith('[')) {
      const mm = s.match(/^\[([^\]]+)\]\(([^)]+)\)$/)!;
      out.push(<a key={i++} href={mm[2]}>{mm[1]}</a>);
    } else out.push(<em key={i++}>{s.slice(1, -1)}</em>);
    last = m.index + s.length;
  }
  if (last < text.length) out.push(<Fragment key={`t${i++}`}>{glossed(text.slice(last), gloss, left, `e${i}`)}</Fragment>);
  return <>{out}</>;
}

export function LangSwitch() {
  const { lang, setLang, t } = useLang();
  return (
    <div className="row" role="group" aria-label={t('language')} style={{ gap: 2 }}>
      {LANGS.map((l) => (
        <button key={l} className="small" aria-pressed={lang === l} lang={l} onClick={() => setLang(l)} aria-label={l === 'en' ? 'English' : 'Slovenčina'} title={l === 'en' ? 'English' : 'Slovenčina'}>
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

