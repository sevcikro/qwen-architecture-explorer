import { useEffect, useRef } from 'react';
import { Cite, Evidence, Tex, fmtInt } from '../components/ui';
import type { InspectorInfo } from './describe';
import { useLang } from '../i18n/i18n';
import type { UiKey } from '../i18n/ui';


export function Inspector({ info, open, onToggle }: { info: InspectorInfo; open: boolean; onToggle: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(open);
  const { t } = useLang();
  useEffect(() => {
    if (open && !wasOpen.current) ref.current?.querySelector<HTMLElement>('h2')?.focus();
    if (!open && wasOpen.current) toggleRef.current?.focus();
    wasOpen.current = open;
  }, [open]);
  return (
    <aside ref={ref} className={`ex-inspector${open ? ' open' : ''}`} aria-label="Inspector">
      <div className="sheet-handle">
        <span className="faint" style={{ fontSize: '0.8rem' }}>{t('in_title')}</span>
        <button ref={toggleRef} className="small sheet-toggle" aria-expanded={open} onClick={onToggle}>{open ? t('in_hide') : `${t('in_details')} ${info.title} ▴`}</button>
      </div>
      <h2 tabIndex={-1} style={{ marginTop: 4 }}>{info.title}</h2>
      <div className="badge-row" style={{ marginBottom: 8 }}>
        <span className="coverage">{t(`role_${info.role}` as UiKey)}</span>
        {info.evidence.map((e, i) => <Evidence key={i} kind={e.kind} title={e.note} />)}
      </div>
      <h3>{t('in_what')}</h3>
      <p>{info.what}</p>
      {info.notes?.map((n, i) => <div key={i} className="callout warn" style={{ fontSize: '0.9rem' }}>{n}</div>)}
      <h3>{t('in_where')}</h3>
      <p>{info.where}</p>
      {(info.inputs || info.outputs) && (
        <>
          <h3>{t('in_io')}</h3>
          <dl className="kv">
            {info.inputs && <><dt>{t('in_enters')}</dt><dd className="mono" style={{ fontSize: '0.85rem' }}>{info.inputs}</dd></>}
            {info.outputs && <><dt>{t('in_leaves')}</dt><dd className="mono" style={{ fontSize: '0.85rem' }}>{info.outputs}</dd></>}
          </dl>
        </>
      )}
      <h3>{t('in_params')}</h3>
      <p>{info.params ? (info.params.count === 0 ? t('in_none') : <><strong>{fmtInt(info.params.count)}</strong> <span className="mono faint" style={{ fontSize: '0.82rem' }}>({info.params.formula})</span></>) : t('in_none_op')}</p>
      <h3>{t('in_persist')}</h3>
      <p>{info.persistence}</p>
      {info.equation && (<><h3>{t('in_op')}</h3><Tex block>{info.equation}</Tex></>)}
      {(info.checkpoint || info.implementation) && (
        <>
          <h3>{t('in_ckpt')}</h3>
          {info.checkpoint?.map((c) => <div key={c}><code>{c}</code></div>)}
          {info.implementation && <div className="faint" style={{ fontSize: '0.85rem', marginTop: 4 }}>{t('in_impl')} <code>{info.implementation}</code></div>}
        </>
      )}
      <h3>{t('in_evidence')}</h3>
      <ul style={{ paddingLeft: 18, fontSize: '0.88rem' }}>
        {info.evidence.map((e, i) => <li key={i}><Evidence kind={e.kind} /> {e.note}</li>)}
      </ul>
      {info.sources.length > 0 && <div style={{ fontSize: '0.85rem' }}>{t('in_sources')}<Cite ids={[...new Set(info.sources)]} /></div>}
      <h3 style={{ marginTop: 12 }}>{t('in_next')}</h3>
      <div className="row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
        {info.below.map((l) => <a key={l.href} className="btn" href={l.href}>↓ {l.label}</a>)}
        {info.above && <a className="btn ghost" href={info.above.href}>↑ {info.above.label}</a>}
      </div>
    </aside>
  );
}
