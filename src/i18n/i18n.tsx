import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { UI, type UiKey } from './ui';

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
export function Rich({ text }: { text: string }) {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const s = m[0];
    if (s.startsWith('**')) out.push(<strong key={i++}>{s.slice(2, -2)}</strong>);
    else if (s.startsWith('`')) out.push(<code key={i++}>{s.slice(1, -1)}</code>);
    else if (s.startsWith('[')) {
      const mm = s.match(/^\[([^\]]+)\]\(([^)]+)\)$/)!;
      out.push(<a key={i++} href={mm[2]}>{mm[1]}</a>);
    } else out.push(<em key={i++}>{s.slice(1, -1)}</em>);
    last = m.index + s.length;
  }
  if (last < text.length) out.push(text.slice(last));
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

/** Shown in Slovak mode on detailed technical pages whose body text is still English. */
export function EnglishOnlyNotice() {
  const { lang, t } = useLang();
  if (lang !== 'sk') return null;
  return (
    <div className="callout" lang="sk" role="note" style={{ fontSize: '0.9rem' }}>
      {t('english_only')} <a href="#/learn">{t('english_only_link')}</a>
    </div>
  );
}
