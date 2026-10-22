import { useEffect, useState } from 'react';
import { CHAPTERS, type Block } from '../content/chapters';
import { Rich, useLang } from '../i18n/i18n';
import { Tex } from '../components/ui';
import { Widget } from '../learn/widgets';

const KEY = 'qa-learn-done';
function loadDone(): string[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '[]'); } catch { return []; }
}
function saveDone(d: string[]) {
  try { localStorage.setItem(KEY, JSON.stringify(d)); } catch { /* storage unavailable: progress lives for this visit only */ }
}

function Quiz({ b }: { b: Extract<Block, { k: 'quiz' }> }) {
  const { t, tr } = useLang();
  const [pickd, setPick] = useState<number | null>(null);
  return (
    <div className="card" style={{ margin: '14px 0', borderLeft: '4px solid var(--focus)' }}>
      <strong>❓ {t('learn_check')}</strong>
      <p style={{ margin: '6px 0' }}><Rich text={tr(b.q)} /></p>
      <div className="row" role="radiogroup" aria-label={tr(b.q)} style={{ flexDirection: 'column', alignItems: 'stretch' }}>
        {b.options.map((o, i) => (
          <button key={i} role="radio" aria-checked={pickd === i} onClick={() => setPick(i)} style={{ justifyContent: 'flex-start', textAlign: 'left', borderColor: pickd === i ? (i === b.correct ? 'var(--ok)' : 'var(--danger)') : undefined }}>
            {pickd === i ? (i === b.correct ? '✓ ' : '✗ ') : ''}{tr(o)}
          </button>
        ))}
      </div>
      {pickd !== null && (
        <p role="status" style={{ marginTop: 8 }}>
          <strong style={{ color: pickd === b.correct ? 'var(--ok)' : 'var(--danger)' }}>{pickd === b.correct ? t('learn_correct') : t('learn_wrong')}</strong> <Rich text={tr(b.explain)} />
        </p>
      )}
    </div>
  );
}

function BlockView({ b }: { b: Block }) {
  const { t, tr } = useLang();
  switch (b.k) {
    case 'p': return <p style={{ fontSize: '1.03rem' }}><Rich text={tr(b.t)} /></p>;
    case 'h': return <h3 style={{ marginTop: 18 }}>{tr(b.t)}</h3>;
    case 'key': return <div className="callout answer"><strong>💡 {t('learn_key')}:</strong> <Rich text={tr(b.t)} /></div>;
    case 'analogy': return <div className="callout" style={{ borderLeftColor: 'var(--norm)' }}><strong>🔎 {t('learn_analogy')}:</strong> <Rich text={tr(b.t)} /></div>;
    case 'qwen': return <div className="callout" style={{ borderLeftColor: 'var(--weight)' }}><strong>📐 {t('learn_qwen')}:</strong> <Rich text={tr(b.t)} /></div>;
    case 'list': return <ul style={{ paddingLeft: 20 }}>{b.items.map((x, i) => <li key={i}><Rich text={tr(x)} /></li>)}</ul>;
    case 'math': return <div style={{ margin: '8px 0' }}><Tex block>{b.tex}</Tex>{b.t && <div className="faint" style={{ fontSize: '0.85rem', textAlign: 'center' }}>{tr(b.t)}</div>}</div>;
    case 'widget': return <Widget id={b.w} />;
    case 'quiz': return <Quiz b={b} />;
    case 'links': return (
      <div style={{ marginTop: 14 }}>
        <strong>{t('learn_deeper')}:</strong>
        <div className="row" style={{ marginTop: 6 }}>{b.items.map((l) => <a key={l.href} className="btn" href={l.href}>{tr(l.label)} →</a>)}</div>
      </div>
    );
  }
}

export default function Learn({ id }: { id?: string }) {
  const { t, tr, lang } = useLang();
  const [done, setDone] = useState<string[]>(loadDone);
  useEffect(() => saveDone(done), [done]);
  const ch = CHAPTERS.find((c) => c.id === id);

  if (!ch) {
    const nextUp = CHAPTERS.find((c) => !done.includes(c.id)) ?? CHAPTERS[0];
    return (
      <div className="page">
        <h1>{t('learn_title')}</h1>
        <p className="muted" style={{ fontSize: '1.05rem' }}>{t('learn_intro')}</p>
        <div className="row" style={{ margin: '10px 0 16px' }}>
          <a className="btn primary" href={`#/learn/${nextUp.id}`} data-testid="learn-start">{done.length ? `${t('learn_continue')}: ${nextUp.n}. ${tr(nextUp.title)}` : t('learn_start')}</a>
          <span className="faint">{done.length} / {CHAPTERS.length} {t('learn_progress')}</span>
        </div>
        <ol style={{ listStyle: 'none', padding: 0 }}>
          {CHAPTERS.map((c) => (
            <li key={c.id} className="card" style={{ marginBottom: 8 }}>
              <a href={`#/learn/${c.id}`} style={{ textDecoration: 'none', color: 'inherit', display: 'block' }}>
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <span className="faint mono">{t('learn_chapter')} {c.n} · {c.minutes} {t('learn_min')}</span>
                  {done.includes(c.id) && <span style={{ color: 'var(--ok)' }} aria-label="done">✓</span>}
                </div>
                <h3 style={{ margin: '2px 0', color: 'var(--accent)' }}>{tr(c.title)}</h3>
                <div className="muted" style={{ fontSize: '0.92rem' }}>{tr(c.summary)}</div>
              </a>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  const i = CHAPTERS.indexOf(ch);
  const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
  const isDone = done.includes(ch.id);
  return (
    <div className="page learn-layout" lang={lang}>
      <aside className="learn-toc" aria-label={t('learn_toc')}>
        <a href="#/learn">{t('learn_all')}</a>
        <ol>
          {CHAPTERS.map((c) => (
            <li key={c.id}><a href={`#/learn/${c.id}`} aria-current={c.id === ch.id ? 'page' : undefined}>{done.includes(c.id) ? '✓ ' : ''}{c.n}. {tr(c.title)}</a></li>
          ))}
        </ol>
      </aside>
      <article>
        <div className="faint mono">{t('learn_chapter')} {ch.n} / {CHAPTERS.length} · {ch.minutes} {t('learn_min')}</div>
        <h1 style={{ marginTop: 4 }}>{tr(ch.title)}</h1>
        <p className="muted" style={{ fontSize: '1.08rem' }}>{tr(ch.summary)}</p>
        <div className="progress" aria-hidden="true"><span style={{ width: `${((i + 1) / CHAPTERS.length) * 100}%` }} /></div>
        {ch.blocks.map((b, k) => <BlockView key={`${ch.id}-${k}`} b={b} />)}
        <div className="row" style={{ marginTop: 22, justifyContent: 'space-between' }}>
          {prev ? <a className="btn" href={`#/learn/${prev.id}`}>{t('learn_prev')}</a> : <span />}
          <button aria-pressed={isDone} onClick={() => setDone(isDone ? done.filter((x) => x !== ch.id) : [...done, ch.id])}>{isDone ? t('learn_undone') : t('learn_done')}</button>
          {next ? <a className="btn primary" href={`#/learn/${next.id}`} onClick={() => !isDone && setDone([...done, ch.id])}>{t('learn_next')} {next.n}. {tr(next.title)}</a> : <a className="btn primary" href="#/explore?m=qwen3-8b&lv=5">{t('nav_explore')} →</a>}
        </div>
        {!next && <p className="callout answer" style={{ marginTop: 12 }}>{t('learn_finished')}</p>}
      </article>
    </div>
  );
}
