import { TOURS } from '../content/tours';
import { useLang } from '../i18n/i18n';

export default function Tours({ id }: { id?: string }) {
  const { t, tr } = useLang();
  const tour = TOURS.find((x) => x.id === id);
  if (!tour) {
    return (
      <div className="page">
        <h1>{t('tours_title')}</h1>
        {id && <div className="callout warn">{t('tours_none')}</div>}
        <div className="grid-auto">
          {TOURS.map((x) => (
            <a key={x.id} className="card" href={`#/tours/${x.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
              <h3 style={{ marginTop: 0, color: 'var(--accent)' }}>{tr(x.title)}</h3>
              <p className="muted">{tr(x.summary)}</p>
              <span className="faint">{x.steps.length} {t('tours_steps')}</span>
            </a>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="page">
      <a href="#/tours">{t('tours_all')}</a>
      <h1>{tr(tour.title)}</h1>
      <p className="muted">{tr(tour.summary)} {t('tours_intro')}</p>
      <ol style={{ paddingLeft: 0, listStyle: 'none' }}>
        {tour.steps.map((s, i) => (
          <li key={i} className="card" style={{ marginBottom: 10 }}>
            <div className="faint mono">{t('tours_step')} {i + 1} {t('tours_of')} {tour.steps.length}</div>
            <h3 style={{ margin: '2px 0 4px' }}>{tr(s.title)}</h3>
            <p style={{ margin: '0 0 8px' }}>{tr(s.text)}</p>
            <a className="btn" href={s.href}>{tr(s.cta)} →</a>
          </li>
        ))}
      </ol>
      <div className="row">
        {TOURS.filter((x) => x.id !== tour.id).map((x) => <a key={x.id} href={`#/tours/${x.id}`}>{t('tours_next')} {tr(x.title)}</a>)}
      </div>
    </div>
  );
}
