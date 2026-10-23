import { useMemo, useState, type ReactNode } from 'react';
import { CODE, CODE_ENV, REPO_URL, type CodeId } from '../content/code';
import { Rich, useLang } from '../i18n/i18n';
import { Evidence } from '../components/ui';

// A tiny, dependency-free Python highlighter: comments, strings, keywords, numbers, builtins, def/class names.
const KW = new Set(['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']);
const BUILTIN = new Set(['print', 'range', 'len', 'sum', 'max', 'min', 'abs', 'round', 'float', 'int', 'list', 'dict', 'set', 'sorted', 'zip', 'enumerate', 'open', 'super', 'isinstance', 'self']);
const TOKEN = /("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfu]{0,2}"(?:\\.|[^"\\\n])*"|[rbfu]{0,2}'(?:\\.|[^'\\\n])*'|#[^\n]*|\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_]\w*\b)/g;

export function highlightPython(src: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0, k = 0, prevWord = '';
  for (const m of src.matchAll(TOKEN)) {
    const t = m[0], i = m.index ?? 0;
    if (i > last) out.push(src.slice(last, i));
    let cls = '';
    if (t.startsWith('#')) cls = 'c';
    else if (/^[rbfu]{0,2}["']/.test(t)) cls = 's';
    else if (/^\d/.test(t)) cls = 'n';
    else if (KW.has(t)) cls = 'k';
    else if (prevWord === 'def' || prevWord === 'class') cls = 'f';
    else if (BUILTIN.has(t)) cls = 'b';
    out.push(cls ? <span key={k++} className={`py-${cls}`}>{t}</span> : t);
    if (/^[A-Za-z_]/.test(t)) prevWord = t;
    last = i + t.length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

const FOLD = 34;

export function CodeBlock({ id, t }: { id: CodeId; t?: string }) {
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const ex = CODE[id];
  const lines = ex.source.replace(/\n$/, '').split('\n');
  const long = lines.length > FOLD;
  const [open, setOpen] = useState(!long);
  const [copied, setCopied] = useState(false);
  const shown = open ? lines.join('\n') : lines.slice(0, FOLD - 6).join('\n');
  const hl = useMemo(() => highlightPython(shown), [shown]);
  return (
    <figure className="card code-card" style={{ margin: '14px 0' }}>
      <div className="code-head">
        <strong>🐍 {T('Python', 'Python')}: <span className="mono">{ex.file}</span></strong>
        <span className="code-actions">
          <button className="small" onClick={() => { navigator.clipboard?.writeText(ex.source).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1600); }, () => setCopied(false)); }}>{copied ? T('Copied', 'Skopírované') : T('Copy code', 'Kopírovať kód')}</button>
          <a className="btn small" href={`https://colab.research.google.com/github/sevcikro/qwen-architecture-explorer/blob/main/notebooks/${ex.file.replace(/\.py$/, '.ipynb')}`} target="_blank" rel="noreferrer" title={T('Run this example in your browser with Google Colab (free Google account needed)', 'Spustiť tento príklad v prehliadači cez Google Colab (treba bezplatný účet Google)')}>{T('Open in Colab', 'Otvoriť v Colabe')}</a>
          <a className="btn small" href={`${REPO_URL}/blob/main/examples/python/${ex.file}`} target="_blank" rel="noreferrer">{T('On GitHub', 'Na GitHube')}</a>
        </span>
      </div>
      {t && <p style={{ margin: '4px 0 8px' }}><Rich text={t} /></p>}
      <div className="faint" style={{ fontSize: '0.8rem', marginBottom: 4 }}>{T('Needs', 'Potrebuje')}: <code>{ex.needs}</code>{ex.realModel && <> · {T('uses the real Qwen3-0.6B files', 'používa skutočné súbory Qwen3-0.6B')}</>}</div>
      <pre className="code" tabIndex={0} aria-label={T(`Python source of ${ex.file}`, `Zdrojový kód ${ex.file}`)}><code>{hl}</code></pre>
      {long && <button className="small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? T('Show less', 'Zobraziť menej') : T(`Show all ${lines.length} lines`, `Zobraziť všetkých ${lines.length} riadkov`)}</button>}
      {ex.ran && ex.output !== null ? (
        <div className="code-out">
          <div className="row" style={{ justifyContent: 'space-between', gap: 6 }}>
            <span className="faint" style={{ fontSize: '0.8rem' }}>{T('Output of a recorded run', 'Výstup zaznamenaného behu')} · Python {CODE_ENV.python} · numpy {CODE_ENV.numpy}{id === 'backprop' || id === 'train-lm' || ex.realModel ? ` · torch ${CODE_ENV.torch}` : ''}{ex.realModel ? ` · transformers ${CODE_ENV.transformers}` : ''} · {CODE_ENV.device} · {CODE_ENV.date}</span>
            {ex.realModel && <Evidence kind="sample" title={T('Real output of the real Qwen3-0.6B files', 'Skutočný výstup zo skutočných súborov Qwen3-0.6B')} />}
          </div>
          <pre className="code out" tabIndex={0} aria-label={T('Program output', 'Výstup programu')}>{ex.output}</pre>
        </div>
      ) : (
        <p className="faint" style={{ fontSize: '0.82rem' }}>{T('Not executed when this site was built: it needs a running model server (see the comment at the top of the file).', 'Pri zostavení stránky sa nespúšťal: potrebuje bežiaci server s modelom (pozrite komentár na začiatku súboru).')}</p>
      )}
    </figure>
  );
}
