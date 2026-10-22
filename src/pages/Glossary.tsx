import { useState } from 'react';
import { GLOSSARY } from '../content/glossary';

export default function Glossary() {
  const [q, setQ] = useState('');
  const items = GLOSSARY.filter((g) => !q || `${g.term} ${g.def}`.toLowerCase().includes(q.toLowerCase())).sort((a, b) => a.term.localeCompare(b.term));
  return (
    <div className="page">
      <h1>Glossary</h1>
      <label className="field" style={{ maxWidth: 420 }}>Search terms<input type="search" value={q} onChange={(e) => setQ(e.target.value)} /></label>
      <dl style={{ marginTop: 12 }}>
        {items.map((g) => (
          <div key={g.term} className="card tight" style={{ marginBottom: 8 }}>
            <dt><strong>{g.term}</strong></dt>
            <dd style={{ margin: '4px 0 0' }}>{g.def}</dd>
          </div>
        ))}
        {items.length === 0 && <p className="muted">No term matches “{q}”.</p>}
      </dl>
    </div>
  );
}
