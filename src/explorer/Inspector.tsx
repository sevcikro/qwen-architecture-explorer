import { useEffect, useRef } from 'react';
import { Cite, Evidence, Tex, fmtInt } from '../components/ui';
import type { InspectorInfo } from './describe';

const ROLE_TEXT: Record<string, string> = {
  param: 'Holds learned parameters',
  norm: 'Normalization with a learned gain',
  act: 'Runtime activation',
  state: 'Runtime cache / state',
  op: 'Stateless operation (no learned weights)',
  group: 'Container',
};

export function Inspector({ info, open, onToggle }: { info: InspectorInfo; open: boolean; onToggle: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(open);
  useEffect(() => {
    if (open && !wasOpen.current) ref.current?.querySelector<HTMLElement>('h2')?.focus();
    if (!open && wasOpen.current) toggleRef.current?.focus();
    wasOpen.current = open;
  }, [open]);
  return (
    <aside ref={ref} className={`ex-inspector${open ? ' open' : ''}`} aria-label="Inspector">
      <div className="sheet-handle">
        <span className="faint" style={{ fontSize: '0.8rem' }}>Inspector</span>
        <button ref={toggleRef} className="small sheet-toggle" aria-expanded={open} onClick={onToggle}>{open ? 'Hide details ▾' : `Details: ${info.title} ▴`}</button>
      </div>
      <h2 tabIndex={-1} style={{ marginTop: 4 }}>{info.title}</h2>
      <div className="badge-row" style={{ marginBottom: 8 }}>
        <span className="coverage">{ROLE_TEXT[info.role]}</span>
        {info.evidence.map((e, i) => <Evidence key={i} kind={e.kind} title={e.note} />)}
      </div>
      <h3>What is this?</h3>
      <p>{info.what}</p>
      {info.notes?.map((n, i) => <div key={i} className="callout warn" style={{ fontSize: '0.9rem' }}>{n}</div>)}
      <h3>Where is it?</h3>
      <p>{info.where}</p>
      {(info.inputs || info.outputs) && (
        <>
          <h3>In → out</h3>
          <dl className="kv">
            {info.inputs && <><dt>enters</dt><dd className="mono" style={{ fontSize: '0.85rem' }}>{info.inputs}</dd></>}
            {info.outputs && <><dt>leaves</dt><dd className="mono" style={{ fontSize: '0.85rem' }}>{info.outputs}</dd></>}
          </dl>
        </>
      )}
      <h3>Learned parameters?</h3>
      <p>{info.params ? (info.params.count === 0 ? 'None.' : <><strong>{fmtInt(info.params.count)}</strong> <span className="mono faint" style={{ fontSize: '0.82rem' }}>({info.params.formula})</span></>) : 'None: this step has no trainable weights.'}</p>
      <h3>Weights or activations?</h3>
      <p>{info.persistence}</p>
      {info.equation && (<><h3>Operation</h3><Tex block>{info.equation}</Tex></>)}
      {(info.checkpoint || info.implementation) && (
        <>
          <h3>Checkpoint tensor / code</h3>
          {info.checkpoint?.map((c) => <div key={c}><code>{c}</code></div>)}
          {info.implementation && <div className="faint" style={{ fontSize: '0.85rem', marginTop: 4 }}>Implementation symbol: <code>{info.implementation}</code></div>}
        </>
      )}
      <h3>Evidence</h3>
      <ul style={{ paddingLeft: 18, fontSize: '0.88rem' }}>
        {info.evidence.map((e, i) => <li key={i}><Evidence kind={e.kind} /> {e.note}</li>)}
      </ul>
      {info.sources.length > 0 && <div style={{ fontSize: '0.85rem' }}>Sources:<Cite ids={[...new Set(info.sources)]} /></div>}
      <h3 style={{ marginTop: 12 }}>Explore next</h3>
      <div className="row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
        {info.below.map((l) => <a key={l.href} className="btn" href={l.href}>↓ {l.label}</a>)}
        {info.above && <a className="btn ghost" href={info.above.href}>↑ {info.above.label}</a>}
      </div>
    </aside>
  );
}
