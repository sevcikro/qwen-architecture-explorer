import type { ReactNode } from 'react';
import { useLang, type Lang } from '../i18n/i18n';
import { Evidence } from '../components/ui';
import type { EvidenceClass } from '../evidence/types';

// Shared building blocks of the course experiments.

export const pick = (lang: Lang, en: string, sk: string) => (lang === 'sk' ? sk : en);

export function Frame({ title, ev, children, note }: { title: string; ev: EvidenceClass[]; children: ReactNode; note?: string }) {
  const { lang } = useLang();
  return (
    <figure className="card widget" style={{ margin: '14px 0', borderColor: 'var(--line)' }}>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 6 }}>
        <strong>🧪 {pick(lang, 'Try it', 'Vyskúšajte')}: {title}</strong>
        <span className="badge-row">{ev.map((e) => <Evidence key={e} kind={e} />)}</span>
      </div>
      {children}
      {note && <figcaption className="faint" style={{ fontSize: '0.82rem', marginTop: 8 }}>{note}</figcaption>}
    </figure>
  );
}

export function Range({ label, value, min, max, step = 0.1, onChange, fmt }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; fmt?: (v: number) => string }) {
  const { num } = useLang();
  return (
    <label className="field" style={{ minWidth: 130, flex: 1 }}>
      <span>{label}: <strong className="num">{fmt ? fmt(value) : num(value, 2)}</strong></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function HBars({ items, max }: { items: { label: string; v: number; on?: boolean; extra?: string }[]; max?: number }) {
  const m = max ?? Math.max(1e-9, ...items.map((x) => Math.abs(x.v)));
  return (
    <div style={{ display: 'grid', gap: 4 }}>
      {items.map((x, i) => (
        <div key={i} className="prob-row" style={{ gridTemplateColumns: '92px minmax(0,1fr) 76px' }}>
          <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{x.label}</span>
          <span className="track" aria-hidden="true"><span className={`fill${x.on === false ? ' off' : ''}`} style={{ width: `${Math.min(100, (Math.abs(x.v) / m) * 100)}%`, display: 'block', background: x.v < 0 ? 'var(--danger)' : undefined }} /></span>
          <span className="mono num" style={{ fontSize: '0.8rem' }}>{x.extra ?? x.v.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

