import { useEffect, useMemo, useState } from 'react';
import { ALL_CORE } from '../models/profiles';
import { tensorName } from '../models/shapes';
import { EvidenceLegend, ExportButton, RoleLegend, Select } from '../components/ui';
import { LEVELS, explorerFromQuery, explorerHref, normalizeExplorer, type ExplorerState, type Route } from '../state/route';
import { describeElement, famByKey, type InspectorInfo } from './describe';
import { GraphCanvas } from './Canvas';
import { blockGraph, moduleGraph, modelGraph, type NodeSpec } from './graph';
import { ExecutionView, MatrixView, ScalarView, UnitView } from './DeepViews';
import { Inspector } from './Inspector';
import { Outline } from './Outline';
import { useLang } from '../i18n/i18n';

const MOD_LABEL: Record<string, string> = { embedding: 'Embedding', attn_norm: 'RMSNorm (attn)', attention: 'Attention', residual1: 'Residual +', ffn_norm: 'RMSNorm (FFN)', ffn: 'SwiGLU FFN', moe: 'MoE', residual2: 'Residual +', final_norm: 'Final norm', lm_head: 'LM head', sampling: 'Sampling' };

export function Explorer({ route, nav }: { route: Route; nav: (href: string, replace?: boolean) => void }) {
  const { t, lang } = useLang();
  const raw = useMemo(() => explorerFromQuery(route.query), [route.query]);
  const { state: s, profile: p, warnings } = useMemo(() => normalizeExplorer(raw, lang), [raw, lang]);
  const [focus, setFocus] = useState<string | undefined>();
  const [compressed, setCompressed] = useState(false);
  const [outline, setOutline] = useState(false);
  const [sheet, setSheet] = useState(false);
  const [notice, setNotice] = useState<string[]>([]);

  useEffect(() => setFocus(undefined), [s.lv, s.m, s.b, s.mod]);
  useEffect(() => {
    if (warnings.length) {
      setNotice(warnings);
      nav(explorerHref(s), true); // canonicalize the URL to the valid view
    }
  }, [warnings.join('|')]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (patch: Partial<ExplorerState>) => {
    const next = { ...s, ...patch };
    for (const k of Object.keys(patch) as (keyof ExplorerState)[]) if (patch[k] === undefined) delete next[k];
    setNotice([]);
    nav(explorerHref(next));
  };
  const openNode = (n: NodeSpec) => n.open && (setNotice([]), nav(explorerHref({ ...n.open, m: p.id })));
  const switchModel = (m: string) => {
    const { state, warnings: w } = normalizeExplorer({ ...s, m }, lang);
    const nm = ALL_CORE.find((x) => x.id === m)?.name;
    setNotice(w.length ? [`${t('ex_switched')} ${nm}. ${t('ex_translated')}`, ...w] : [`${t('ex_switched')} ${nm}; ${t('ex_switched_kept')}`]);
    nav(explorerHref(state));
  };

  const fam = s.t ? famByKey(p, s.t) : undefined;
  const graph = s.lv === 5 ? modelGraph(p, { compressed, selectedBlock: s.b }, lang) : s.lv === 4 ? blockGraph(p, s.b ?? 0, lang) : s.lv === 3 ? moduleGraph(p, s.mod ?? 'attention', s.b ?? 0, s.e, lang) : null;
  const defaultId = s.lv === 5 ? (s.b !== undefined ? `block:${s.b}` : 'model') : s.lv === 4 ? `block:${s.b}` : s.lv === 3 ? (s.mod === 'moe' && s.e !== undefined ? `expert:${s.e}` : s.mod ?? 'model') : s.t ? `tensor:${s.t}` : 'model';
  const info: InspectorInfo = describeElement(p, focus ?? defaultId, s, lang);
  const selectNode = (id: string) => {
    setFocus(id);
    if (id.startsWith('block:') && s.lv === 5) nav(explorerHref({ ...s, b: Number(id.slice(6)) }), true);
    if (id.startsWith('expert:')) nav(explorerHref({ ...s, e: Number(id.slice(7)) }), true);
    if (id.startsWith('gap:')) setCompressed(false);
  };

  // breadcrumbs
  const crumbs: { label: string; to: Partial<ExplorerState>; lv: number }[] = [{ label: p.name, to: { lv: 5, b: s.b }, lv: 5 }];
  if (s.b !== undefined && s.lv <= 4) crumbs.push({ label: `Block ${s.b}`, to: { lv: 4, b: s.b, mod: undefined, t: undefined, r: undefined, c: undefined }, lv: 4 });
  if (s.mod && s.lv <= 3) crumbs.push({ label: MOD_LABEL[s.mod] ?? s.mod, to: { lv: 3, t: undefined, r: undefined, c: undefined }, lv: 3 });
  if (s.e !== undefined && s.mod === 'moe' && s.lv <= 2) crumbs.push({ label: `Expert ${s.e}`, to: { lv: 3, t: undefined, r: undefined, c: undefined }, lv: 3 });
  if (fam && s.lv <= 2) crumbs.push({ label: fam.label.split(' ')[0] === 'Expert' ? fam.label : fam.key, to: { lv: 2, r: s.lv < 2 ? s.r : undefined, c: s.lv < 2 ? s.c : undefined }, lv: 2 });
  if (fam && s.r !== undefined && s.lv <= 1 && fam.shape.length > 1) crumbs.push({ label: `row ${s.r}`, to: { lv: 1 }, lv: 1 });
  if (fam && s.lv === 0) crumbs.push({ label: `[${s.r}${s.c !== undefined ? `, ${s.c}` : ''}]`, to: { lv: 0 }, lv: 0 });
  if (s.lv === 6) crumbs.push({ label: 'Execution & learning', to: { lv: 6 }, lv: 6 });

  const levelAvailable = (lv: number) => {
    if (lv >= 5) return true;
    if (lv === 4) return s.b !== undefined || p.layers > 0;
    if (lv === 3) return true;
    if (lv === 2) return !!fam;
    if (lv === 1) return !!fam && fam.shape.length > 1;
    if (lv === 0) return !!fam;
    return false;
  };
  const toLevel = (lv: number) => {
    if (lv === 4) go({ lv, b: s.b ?? 0 });
    else if (lv === 3) go({ lv, b: s.mod && ['embedding', 'final_norm', 'lm_head', 'sampling'].includes(s.mod) ? undefined : s.b ?? 0, mod: s.mod ?? 'attention' });
    else if (lv === 1) go({ lv, r: s.r ?? 0 });
    else if (lv === 0) go({ lv, r: s.r ?? 0, c: fam && fam.shape.length > 1 ? s.c ?? 0 : undefined });
    else go({ lv });
  };
  const parent = crumbs.length > 1 ? crumbs[crumbs.length - 2] : null;

  return (
    <div className="explorer">
      <div className="ex-bar">
        <Select inline label={t('ex_model')} value={p.id} onChange={switchModel} options={ALL_CORE.map((x) => ({ value: x.id, label: x.status === 'toy' ? `${x.name}` : `${x.name} (${t('ex_verified_structure')})` }))} />
        <span className="coverage">{p.status === 'toy' ? t('ex_cov_toy') : t('ex_cov_full')}</span>
        <div className="levels" role="toolbar" aria-label={t('ex_level_bar')}>
          {[...LEVELS].reverse().map((L) => (
            <button key={L.lv} className="small" aria-pressed={s.lv === L.lv} disabled={!levelAvailable(L.lv)} title={levelAvailable(L.lv) ? L.long : `${L.long}: ${t('ex_need_tensor')}`} onClick={() => toLevel(L.lv)}>{L.lv} {t(`lv${L.lv}` as 'lv0')}</button>
          ))}
        </div>
        <nav className="crumbs" aria-label={t('ex_breadcrumb')}>
          {crumbs.map((c, i) => (
            <span key={i} className="row" style={{ gap: 2 }}>
              {i > 0 && <span className="sep" aria-hidden="true">›</span>}
              <button aria-current={i === crumbs.length - 1 ? 'true' : undefined} onClick={() => go(c.to)}>{c.label}</button>
            </span>
          ))}
        </nav>
        <span className="spacer" style={{ flex: 1 }} />
        {parent && <button className="small" onClick={() => go(parent.to)}>{t('ex_back')}</button>}
        {s.lv === 5 && <button className="small" aria-pressed={compressed} onClick={() => setCompressed(!compressed)}>{compressed ? t('ex_show_all') : t('ex_compress')}</button>}
        <button className="small" aria-pressed={outline} onClick={() => setOutline(!outline)}>{t('ex_outline')}</button>
        <button className="small" onClick={() => { navigator.clipboard?.writeText(window.location.href).then(() => setNotice([t('ex_copied')]), () => setNotice([`${t('ex_copy_this')} ${window.location.href}`])); }}>{t('ex_share')}</button>
        <ExportButton filename={`qwen-atlas-view-${p.id}.json`} label={t('ex_export')} data={() => ({
          view: s, model: p.repo ?? p.name, revision: p.revision ?? null, status: p.status, tensor: fam ? tensorName(fam, s.b ?? 0, s.e ?? 0) : null, shape: fam?.shape ?? null,
          dtype: p.status === 'official' ? p.dtype : 'float64 (toy)', inspector: { title: info.title, evidence: info.evidence, params: info.params, sources: info.sources },
        })} />
        <button className="small" onClick={() => { setCompressed(false); nav(explorerHref({ m: p.id, lv: 5 })); }} title={t('ex_reset_title')}>{t('ex_reset')}</button>
      </div>
      <div className="ex-stage">
        {notice.length > 0 && (
          <div className="callout warn" role="status" style={{ margin: 8 }}>
            {notice.map((n, i) => <div key={i}>{n}</div>)}
            <button className="small ghost" onClick={() => setNotice([])}>{t('ex_dismiss')}</button>
          </div>
        )}
        {outline && <Outline p={p} s={s} onClose={() => setOutline(false)} />}
        {graph && (
          <>
            <GraphCanvas spec={graph} selected={focus ?? defaultId} onSelect={selectNode} onOpen={openNode} fitKey={`${p.id}|${s.lv}|${s.b}|${s.mod}|${compressed}|${lang}`} />
            <div className="ex-legend" tabIndex={0} style={{ padding: '6px 12px', borderTop: '1px solid var(--line-soft)' }}><RoleLegend /></div>
          </>
        )}
        {!graph && (
          <div className="ex-scroll">
            {s.lv === 6 && <ExecutionView p={p} />}
            {fam && s.lv === 2 && <MatrixView key={`${p.id}${s.t}${s.b}${s.e}`} p={p} f={fam} s={s} go={go} />}
            {fam && s.lv === 1 && <UnitView p={p} f={fam} s={s} go={go} />}
            {fam && s.lv === 0 && <ScalarView key={`${p.id}${s.t}${s.b}${s.e}${s.r}${s.c}`} p={p} f={fam} s={s} go={go} />}
            <div className="page" style={{ paddingTop: 0 }}><EvidenceLegend /></div>
          </div>
        )}
      </div>
      <Inspector info={info} open={sheet} onToggle={() => setSheet(!sheet)} />
    </div>
  );
}
