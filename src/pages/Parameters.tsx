import { useState } from 'react';
import { CORE_PROFILES, type CoreProfile } from '../models/profiles';
import { accounting, familyParams, tensorFamilies, type TensorFamily } from '../models/shapes';
import { Cite, Evidence, ExportButton, Section, Select, fmtB, fmtInt, shapeStr } from '../components/ui';
import { explorerHref, type Route } from '../state/route';
import { EnglishOnlyNotice, useLang } from '../i18n/i18n';

interface Seg { id: string; label: string; value: number; color: string; fams: TensorFamily[]; formula: string }

function segments(p: CoreProfile): Seg[] {
  const f = tensorFamilies(p);
  const by = (keys: string[]) => f.filter((x) => keys.includes(x.key));
  const sum = (fs: TensorFamily[]) => fs.reduce((s, x) => s + familyParams(x), 0);
  const a = accounting(p);
  const segs: Seg[] = [
    { id: 'embed', label: 'Input embedding', value: a.embedding, color: 'var(--weight)', fams: by(['embed_tokens']), formula: `${fmtInt(p.vocabRows)} × ${fmtInt(p.d)}` },
    { id: 'head', label: 'Output head (untied)', value: a.head, color: 'color-mix(in srgb, var(--weight) 70%, var(--bg))', fams: by(['lm_head']), formula: `${fmtInt(p.vocabRows)} × ${fmtInt(p.d)}` },
    { id: 'q', label: `W_Q × ${p.layers}`, value: sum(by(['q_proj'])), color: 'var(--attn)', fams: by(['q_proj']), formula: `${p.layers} × ${fmtInt(p.hq * p.headDim)} × ${fmtInt(p.d)}` },
    { id: 'kv', label: `W_K + W_V × ${p.layers}`, value: sum(by(['k_proj', 'v_proj'])), color: 'color-mix(in srgb, var(--attn) 70%, var(--bg))', fams: by(['k_proj', 'v_proj']), formula: `${p.layers} × 2 × ${fmtInt(p.hkv * p.headDim)} × ${fmtInt(p.d)}` },
    { id: 'o', label: `W_O × ${p.layers}`, value: sum(by(['o_proj'])), color: 'color-mix(in srgb, var(--attn) 55%, var(--bg))', fams: by(['o_proj']), formula: `${p.layers} × ${fmtInt(p.d)} × ${fmtInt(p.hq * p.headDim)}` },
  ];
  if (p.kind === 'dense') {
    segs.push(
      { id: 'gate', label: `W_gate × ${p.layers}`, value: sum(by(['gate_proj'])), color: 'var(--ffn)', fams: by(['gate_proj']), formula: `${p.layers} × ${fmtInt(p.ffnWidth!)} × ${fmtInt(p.d)}` },
      { id: 'up', label: `W_up × ${p.layers}`, value: sum(by(['up_proj'])), color: 'color-mix(in srgb, var(--ffn) 75%, var(--bg))', fams: by(['up_proj']), formula: `${p.layers} × ${fmtInt(p.ffnWidth!)} × ${fmtInt(p.d)}` },
      { id: 'down', label: `W_down × ${p.layers}`, value: sum(by(['down_proj'])), color: 'color-mix(in srgb, var(--ffn) 55%, var(--bg))', fams: by(['down_proj']), formula: `${p.layers} × ${fmtInt(p.d)} × ${fmtInt(p.ffnWidth!)}` },
    );
  } else {
    segs.push(
      { id: 'experts', label: `${p.moe!.numExperts} experts × ${p.layers} blocks`, value: sum(by(['expert_gate_proj', 'expert_up_proj', 'expert_down_proj'])), color: 'var(--ffn)', fams: by(['expert_gate_proj', 'expert_up_proj', 'expert_down_proj']), formula: `${p.layers} × ${p.moe!.numExperts} × 3 × ${fmtInt(p.d)} × ${p.moe!.expertWidth}` },
      { id: 'router', label: `Routers × ${p.layers}`, value: sum(by(['router'])), color: 'var(--router)', fams: by(['router']), formula: `${p.layers} × ${p.moe!.numExperts} × ${fmtInt(p.d)}` },
    );
  }
  segs.push({ id: 'norms', label: 'All norm gains', value: sum(by(['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm', 'norm'])), color: 'var(--norm)', fams: by(['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm', 'norm']), formula: `${p.layers} × (2 × ${fmtInt(p.d)} + 2 × ${p.headDim}) + ${fmtInt(p.d)}` });
  return segs.sort((x, y) => y.value - x.value);
}

/** Squarified treemap layout (Bruls, Huizing & van Wijk) in unit-square coordinates. */
function squarify(values: number[], x: number, y: number, w: number, h: number): { x: number; y: number; w: number; h: number }[] {
  const total = values.reduce((a, b) => a + b, 0);
  const area = values.map((v) => (v / total) * w * h);
  const out: { x: number; y: number; w: number; h: number }[] = new Array(values.length);
  let i = 0;
  let rx = x, ry = y, rw = w, rh = h;
  while (i < area.length) {
    const side = Math.min(rw, rh);
    let row = [i];
    const worst = (r: number[]) => {
      const s = r.reduce((a, k) => a + area[k], 0);
      const mx = Math.max(...r.map((k) => area[k])), mn = Math.min(...r.map((k) => area[k]));
      return Math.max((side * side * mx) / (s * s), (s * s) / (side * side * mn));
    };
    while (i + row.length < area.length && worst([...row, i + row.length]) <= worst(row)) row = [...row, i + row.length];
    const s = row.reduce((a, k) => a + area[k], 0);
    if (rw >= rh) {
      const cw = s / rh;
      let cy = ry;
      for (const k of row) { const ch = area[k] / cw; out[k] = { x: rx, y: cy, w: cw, h: ch }; cy += ch; }
      rx += cw; rw -= cw;
    } else {
      const ch = s / rw;
      let cx = rx;
      for (const k of row) { const cw = area[k] / ch; out[k] = { x: cx, y: ry, w: cw, h: ch }; cx += cw; }
      ry += ch; rh -= ch;
    }
    i += row.length;
  }
  return out;
}

export default function Parameters({ route }: { route: Route }) {
  const initial = CORE_PROFILES.find((x) => x.id === route.query.get('m')) ?? CORE_PROFILES[0];
  const [p, setP] = useState(initial);
  const segs = segments(p);
  const [sel, setSel] = useState<string>(segs[0].id);
  const S = segs.find((x) => x.id === sel) ?? segs[0];
  const { lang } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const rects = squarify(segs.map((s) => s.value), 0, 0, 100, 100);
  const a = accounting(p);
  return (
    <div className="page">
      <h1>{T('Where are the parameters?', 'Kde sú parametre?')}</h1>
      <EnglishOnlyNotice />
      <div className="callout answer">
        {T('The learned parameters are the stored numbers in these matrices and vectors. The changing highlighted values flowing between modules are activations.', 'Naučené parametre sú uložené čísla v týchto maticiach a vektoroch. Meniace sa zvýraznené hodnoty, ktoré tečú medzi modulmi, sú aktivácie.')}{' '}
        <a href={explorerHref({ m: p.id, lv: 0, b: 0, t: 'q_proj', r: 0, c: 0 })}>{T('Show me one actual parameter →', 'Ukážte mi jeden skutočný parameter →')}</a>
      </div>
      <Select label="Model" value={p.id} onChange={(id) => { const n = CORE_PROFILES.find((x) => x.id === id)!; setP(n); setSel(segments(n)[0].id); }} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
      <div className="grid2" style={{ marginTop: 12 }}>
        <div>
          <div className="treemap" role="group" aria-label={`Parameter treemap for ${p.name}; area proportional to parameter count`}>
            {segs.map((s, i) => {
              const r = rects[i];
              return (
                <button key={s.id} className={sel === s.id ? 'sel' : ''} aria-pressed={sel === s.id}
                  style={{ left: `${r.x}%`, top: `${r.y}%`, width: `${r.w}%`, height: `${r.h}%`, background: s.color }}
                  aria-label={`${s.label}: ${fmtInt(s.value)} parameters, ${((s.value / a.total) * 100).toFixed(2)}%`}
                  onClick={() => setSel(s.id)}>
                  {r.w * r.h > 40 && <span className="tl">{s.label}<br />{fmtB(s.value)} · {((s.value / a.total) * 100).toFixed(1)}%</span>}
                </button>
              );
            })}
          </div>
          <p className="faint" style={{ fontSize: '0.82rem' }}>Area ∝ parameter count. Every segment comes from the same shape table used by the diagrams and catalog. <Evidence kind="derived" /></p>
        </div>
        <div className="card">
          <h2 style={{ marginTop: 0 }}>{S.label}</h2>
          <p className="big-num" style={{ fontSize: '1.4rem' }}>{fmtInt(S.value)}</p>
          <p className="mono faint">{S.formula} · {((S.value / a.total) * 100).toFixed(3)}% of the model</p>
          <table className="t"><thead><tr><th>tensor</th><th>shape</th><th className="num">count</th></tr></thead>
            <tbody>{S.fams.map((f) => (
              <tr key={f.key}><td><a href={explorerHref({ m: p.id, lv: 2, t: f.key, b: f.scope === 'model' ? undefined : 0, e: f.scope === 'expert' ? 0 : undefined })}><code>{f.pathTemplate}</code></a></td><td className="mono">{shapeStr(f.shape)}</td><td className="num">{fmtInt(f.multiplicity)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      <Section title={`Exact accounting · ${p.name}`} badges={<><Evidence kind="verified" /><Evidence kind="derived" /></>}>
        <div className="table-wrap"><table className="t">
          <tbody>
            <tr><td>Input embedding</td><td className="num">{fmtInt(a.embedding)}</td></tr>
            <tr><td>Untied output head</td><td className="num">{fmtInt(a.head)}</td></tr>
            <tr><td>Attention projections per block (Q {fmtInt(a.qkvoPerBlock.q)} + K {fmtInt(a.qkvoPerBlock.k)} + V {fmtInt(a.qkvoPerBlock.v)} + O {fmtInt(a.qkvoPerBlock.o)})</td><td className="num">{fmtInt(a.attentionPerBlock)}</td></tr>
            {p.kind === 'moe' && <tr><td>One expert FFN (3 × {fmtInt(p.d)} × {p.moe!.expertWidth})</td><td className="num">{fmtInt(a.oneExpert!)}</td></tr>}
            <tr><td>All FFN / expert weights per block</td><td className="num">{fmtInt(a.ffnPerBlock)}</td></tr>
            <tr><td>Router weights per block</td><td className="num">{fmtInt(a.routerPerBlock)}</td></tr>
            <tr><td>Norm gains per block (2 × {fmtInt(p.d)} + 2 × {p.headDim})</td><td className="num">{fmtInt(a.normsPerBlock)}</td></tr>
            <tr><td>Complete parameters per block</td><td className="num">{fmtInt(a.perBlock)}</td></tr>
            <tr><td>Final norm</td><td className="num">{fmtInt(a.finalNorm)}</td></tr>
            <tr><td><strong>TOTAL</strong> (= header element count of the pinned checkpoint)</td><td className="num"><strong>{fmtInt(a.total)}</strong></td></tr>
          </tbody>
        </table></div>
        <ExportButton filename={`${p.id}-parameter-accounting.json`} data={() => ({ model: p.repo, revision: p.revision, evidence: 'derived from verified config shapes; total cross-checked against checkpoint headers', convention: 'logical tensors, each scalar counted once; no biases; untied embeddings', accounting: a })} />
        <h3>Exact count vs advertised figure</h3>
        <p>The model card says <strong>{p.advertised?.total}</strong>{p.advertised?.nonEmbedding && <> ({p.advertised.nonEmbedding} non-embedding)</>}. The exact structural count is {fmtInt(a.total)} = {(a.total / 1e9).toFixed(4)}B; the card rounds it. The card's “non-embedding” figure matches total − input embedding − output head = {fmtInt(a.nonEmbedding)} ≈ {(a.nonEmbedding / 1e9).toFixed(2)}B: both vocabulary-sized matrices are excluded (subtracting only the input table would give {((a.total - a.embedding) / 1e9).toFixed(2)}B, which does not match).<Cite ids={[p.id === 'qwen3-8b' ? 'q8-card' : 'q30-card']} /></p>
        {p.kind === 'moe' && (
          <>
            <h3>“Activated” parameters depend on a counting convention</h3>
            <ul style={{ paddingLeft: 18 }}>
              <li>Selected experts in all 48 blocks: 48 × 8 × {fmtInt(a.oneExpert!)} = <strong>{fmtInt(a.selectedExpertsAllLayers!)}</strong></li>
              <li>Plus every non-expert parameter (attention, routers, norms, the whole embedding table and head): <strong>{fmtInt(a.activeIncludingFullEmbedding!)}</strong> ≈ 3.35B, consistent with the model card's “3.3B activated”. But a token lookup reads one embedding row, not the table.</li>
              <li>Counting one embedding row instead: {fmtInt(a.activeEmbeddingRowOnly!)}.</li>
            </ul>
            <p className="faint">None of these is a universal exact number of scalars read, FLOPs, memory traffic or latency.</p>
          </>
        )}
      </Section>
    </div>
  );
}
