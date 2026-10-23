import { useState } from 'react';
import { GLOSSARY } from '../content/glossary';
import { useLang } from '../i18n/i18n';

export default function Glossary({ q0 = '' }: { q0?: string }) {
  const { t, tr, lang } = useLang();
  const [q, setQ] = useState(q0);
  const items = GLOSSARY.filter((g) => !q || `${tr(g.term)} ${tr(g.def)} ${g.term.en}`.toLowerCase().includes(q.toLowerCase())).sort((a, b) => tr(a.term).localeCompare(tr(b.term), lang));
  return (
    <div className="page">
      <h1>{t('gl_title')}</h1>
      <label className="field" style={{ maxWidth: 420 }}>{t('gl_search')}<input type="search" value={q} onChange={(e) => setQ(e.target.value)} /></label>
      <dl style={{ marginTop: 12 }}>
        {items.map((g) => (
          <div key={g.term.en} className="card tight" style={{ marginBottom: 8 }}>
            <dt><strong>{tr(g.term)}</strong>{lang === 'sk' && g.term.sk !== g.term.en && <span className="faint" lang="en"> · {g.term.en}</span>}</dt>
            <dd style={{ margin: '4px 0 0' }}>{tr(g.def)}</dd>
          </div>
        ))}
        {items.length === 0 && <p className="muted">{t('gl_none')} {lang === 'sk' ? `„${q}“` : `“${q}”`}.</p>}
      </dl>
    </div>
  );
}
