import { TOURS } from '../content/tours';

export default function Tours({ id }: { id?: string }) {
  const t = TOURS.find((x) => x.id === id);
  if (!t) {
    return (
      <div className="page">
        <h1>Guided tours</h1>
        {id && <div className="callout warn">No tour called “{id}”.</div>}
        <div className="grid-auto">
          {TOURS.map((x) => (
            <a key={x.id} className="card" href={`#/tours/${x.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
              <h3 style={{ marginTop: 0, color: 'var(--accent)' }}>{x.title}</h3>
              <p className="muted">{x.summary}</p>
              <span className="faint">{x.steps.length} steps</span>
            </a>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="page">
      <a href="#/tours">← All tours</a>
      <h1>{t.title}</h1>
      <p className="muted">{t.summary} Each step opens the explorer or a lab in place; use your browser’s Back button to return here.</p>
      <ol style={{ paddingLeft: 0, listStyle: 'none' }}>
        {t.steps.map((s, i) => (
          <li key={i} className="card" style={{ marginBottom: 10 }}>
            <div className="faint mono">Step {i + 1} of {t.steps.length}</div>
            <h3 style={{ margin: '2px 0 4px' }}>{s.title}</h3>
            <p style={{ margin: '0 0 8px' }}>{s.text}</p>
            <a className="btn" href={s.href}>{s.cta} →</a>
          </li>
        ))}
      </ol>
      <div className="row">
        {TOURS.filter((x) => x.id !== t.id).map((x) => <a key={x.id} href={`#/tours/${x.id}`}>Next tour: {x.title}</a>)}
      </div>
    </div>
  );
}
