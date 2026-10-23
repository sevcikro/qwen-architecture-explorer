import { useEffect, useId, useMemo, useState, type ReactNode } from 'react';
import type katexType from 'katex';
import { EVIDENCE_META, type EvidenceClass } from '../evidence/types';
import { sourceById } from '../evidence/sources';
import { useLang } from '../i18n/i18n';

// ---------- number formatting ----------
export const fmtInt = (n: number) => n.toLocaleString('en-US');
export const fmtB = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(3)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));
export function fmtNum(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e4 || a < 1e-3) return x.toExponential(digits - 1);
  return x.toFixed(digits);
}
export const shapeStr = (s: number[]) => `[${s.map(fmtInt).join(', ')}]`;

// ---------- evidence ----------
export function Evidence({ kind, title }: { kind: EvidenceClass; title?: string }) {
  const m = EVIDENCE_META[kind];
  const { t, lang } = useLang();
  return (
    <span className={`ev ev-${kind}`} title={title && lang === 'en' ? title : t(`evd_${kind}`)} data-evidence={kind}>
      <span className="g" aria-hidden="true">{m.glyph}</span>
      {t(`ev_${kind}`)}
    </span>
  );
}

export function SimBanner({ children }: { children?: ReactNode }) {
  const { t } = useLang();
  return (
    <div className="banner-sim" role="note">
      <Evidence kind="simulation" />
      <span><strong>{t('sim_banner')}</strong> {children ?? t('sim_default')}</span>
    </div>
  );
}

export function EvidenceLegend() {
  const { t } = useLang();
  return (
    <div className="legend" aria-label="Evidence classes">
      {(Object.keys(EVIDENCE_META) as EvidenceClass[]).map((k) => (
        <span key={k}><Evidence kind={k} /> <span className="sr-only">{t(`evd_${k}`)}</span></span>
      ))}
    </div>
  );
}

export function RoleLegend() {
  const { t } = useLang();
  return (
    <div className="legend" aria-label="Diagram legend">
      <span><i className="sw param" aria-hidden="true" /> {t('lg_param')}</span>
      <span><i className="sw norm" aria-hidden="true" /> {t('lg_norm')}</span>
      <span><i className="sw act" aria-hidden="true" /> {t('lg_act')}</span>
      <span><i className="sw state" aria-hidden="true" /> {t('lg_state')}</span>
      <span><i className="sw op" aria-hidden="true" /> {t('lg_op')}</span>
    </div>
  );
}

// ---------- sources ----------
export function Cite({ ids }: { ids: string[] }) {
  return (
    <span className="faint" style={{ fontSize: '0.82rem' }}>
      {' '}[
      {ids.map((id, i) => {
        const s = sourceById(id);
        return (
          <span key={id}>
            {i > 0 && '; '}
            <a href={s.url} target="_blank" rel="noreferrer" title={`${s.title}${s.revision ? ` @ ${s.revision.slice(0, 12)}` : ''}`}>{s.title.split(' (')[0]}</a>
          </span>
        );
      })}
      ]
    </span>
  );
}

// ---------- math ----------
let katexLib: typeof katexType | null = null;
let katexLoading: Promise<typeof katexType> | null = null;
const loadKatex = () => (katexLoading ??= import('./katexLoader').then((m) => (katexLib = m.default)));

export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const [ready, setReady] = useState(katexLib !== null);
  useEffect(() => { if (!ready) loadKatex().then(() => setReady(true), () => { /* keep the plain-text fallback */ }); }, [ready]);
  const html = useMemo(() => {
    if (!ready || !katexLib) return null;
    try {
      return katexLib.renderToString(children, { displayMode: block, throwOnError: false, strict: 'ignore', output: 'htmlAndMathml' });
    } catch {
      return null;
    }
  }, [children, block, ready]);
  if (html === null) return block ? <div className="eq mono" tabIndex={0}>{children}</div> : <span className="mono">{children}</span>;
  return block ? <div className="eq" tabIndex={0} dangerouslySetInnerHTML={{ __html: html }} /> : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// ---------- controls ----------
export function NumberField({ label, value, onChange, min, max, step, id, width = 110, inline = false }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; id?: string; width?: number; inline?: boolean }) {
  const gen = useId();
  const fid = id ?? gen;
  return (
    <label className={`field${inline ? ' inline' : ''}`} htmlFor={fid}>
      {label}
      <input
        id={fid}
        type="number"
        value={Number.isFinite(value) ? value : ''}
        min={min}
        max={max}
        step={step ?? 'any'}
        style={{ width }}
        onChange={(e) => {
          const v = e.target.valueAsNumber;
          if (Number.isFinite(v)) onChange(min !== undefined && v < min ? min : max !== undefined && v > max ? max : v);
        }}
      />
    </label>
  );
}

export function Slider({ label, value, onChange, min, max, step = 1, format }: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number; format?: (v: number) => string }) {
  const id = useId();
  return (
    <label className="field" htmlFor={id} style={{ minWidth: 160, flex: 1 }}>
      <span>{label}: <strong className="num">{format ? format(value) : value}</strong></span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Select<T extends string | number>({ label, value, options, onChange, inline = false }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; inline?: boolean }) {
  const id = useId();
  return (
    <label className={`field${inline ? ' inline' : ''}`} htmlFor={id}>
      {label}
      <select
        id={id}
        value={String(value)}
        onChange={(e) => {
          const o = options.find((x) => String(x.value) === e.target.value);
          if (o) onChange(o.value);
        }}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: { tabs: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === value}
          tabIndex={t.id === value ? 0 : -1}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => {
            const i = tabs.findIndex((x) => x.id === value);
            if (e.key === 'ArrowRight') onChange(tabs[(i + 1) % tabs.length].id);
            if (e.key === 'ArrowLeft') onChange(tabs[(i - 1 + tabs.length) % tabs.length].id);
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function VectorView({ values, kind = 'act', highlight, label, digits = 3, max = 32 }: { values: number[]; kind?: 'act' | 'par'; highlight?: number | number[]; label?: string; digits?: number; max?: number }) {
  const hl = new Set(Array.isArray(highlight) ? highlight : highlight === undefined ? [] : [highlight]);
  const shown = values.slice(0, max);
  return (
    <div>
      {label && <div className="faint" style={{ fontSize: '0.8rem' }}>{label}</div>}
      <div className="vec" role="list" aria-label={label}>
        {shown.map((v, i) => (
          <span key={i} role="listitem" className={`v ${kind}${hl.has(i) ? ' hl' : ''}`} title={`[${i}] = ${v}`}>
            {fmtNum(v, digits)}
          </span>
        ))}
        {values.length > max && <span className="faint">… {values.length - max} more</span>}
      </div>
    </div>
  );
}

export function ProbRow({ label, p, off = false, extra }: { label: ReactNode; p: number; off?: boolean; extra?: string }) {
  return (
    <div className="prob-row">
      <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
      <span className="track" aria-hidden="true"><span className={`fill${off ? ' off' : ''}`} style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%`, display: 'block' }} /></span>
      <span className="mono num" style={{ fontSize: '0.8rem' }}>{extra ?? p.toFixed(4)}</span>
    </div>
  );
}

/** Download a JSON export that always carries provenance labels. */
export function ExportButton({ data, filename, label }: { data: () => unknown; filename: string; label?: string }) {
  const { lang } = useLang();
  return (
    <button
      className="small"
      onClick={() => {
        const payload = { exportedBy: 'Qwen Atlas', exportedAt: new Date().toISOString(), ...(data() as object) };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}
    >
      ⭳ {label ?? (lang === 'sk' ? 'Exportovať JSON' : 'Export JSON')}
    </button>
  );
}

export function Section({ title, children, id, badges }: { title: string; children: ReactNode; id?: string; badges?: ReactNode }) {
  return (
    <section className="card" id={id} style={{ marginBottom: 16 }} aria-labelledby={id ? `${id}-h` : undefined}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 id={id ? `${id}-h` : undefined} style={{ marginBottom: 6 }}>{title}</h2>
        {badges && <div className="badge-row">{badges}</div>}
      </div>
      {children}
    </section>
  );
}
