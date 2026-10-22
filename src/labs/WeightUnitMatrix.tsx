import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { forwardFull, toyTensor, toyTensorNames, withEditedWeight } from '../math/toy';
import { maxAbsDiff, dot } from '../math/linalg';
import { silu, sigmoid } from '../math/ops';
import { Evidence, NumberField, Section, Select, VectorView, fmtInt, fmtNum, shapeStr } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { TinyExample } from './TinyExample';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE } from '../models/profiles';
import { tensorFamilies } from '../models/shapes';
import { explorerHref } from '../state/route';
import { useSamples } from '../data/data';
import { locateTensor } from '../explorer/locate';

export function WeightLab() {
  const { profile, weights, state, trace, dispatch } = useToy();
  const names = toyTensorNames(profile);
  const [name, setName] = useState('model.layers.1.self_attn.v_proj.weight');
  const tname = names.includes(name) ? name : names[1];
  const t = toyTensor(weights, tname);
  const isVec = !Array.isArray((t as number[][])[0]);
  const rows = isVec ? (t as number[]).length : (t as number[][]).length;
  const cols = isVec ? 1 : (t as number[][])[0].length;
  const [r, setR] = useState(0);
  const [c, setC] = useState(0);
  const rr = Math.min(r, rows - 1), cc = Math.min(c, cols - 1);
  const cur = isVec ? (t as number[])[rr] : (t as number[][])[rr][cc];
  const [draft, setDraft] = useState<number | null>(null);
  const eff = useMemo(() => {
    if (draft === null || !Number.isFinite(draft)) return null;
    const w2 = withEditedWeight(weights, tname, rr, isVec ? null : cc, draft);
    const tr2 = forwardFull(profile, w2, state.tokens);
    const layer = tname.match(/layers\.(\d+)/)?.[1];
    const perLayer = trace[0].layers.map((_, l) => Math.max(...trace.map((p, i) => maxAbsDiff(p.layers[l].out, tr2[i].layers[l].out))));
    return { tr2, layer: layer === undefined ? null : Number(layer), perLayer, logitDiff: Math.max(...trace.map((p, i) => maxAbsDiff(p.logits, tr2[i].logits))) };
  }, [draft, weights, tname, rr, cc, isVec, profile, state.tokens, trace]);
  return (
    <>
      <Section title="1 · The tiny example" badges={<Evidence kind="simulation" />}>
        <TinyExample />
      </Section>
      <Section title="Weight microscope: change one stored number in the toy model" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <div className="row">
          <Select label="Toy tensor (Qwen-style name)" value={tname} onChange={(v) => { setName(v); setR(0); setC(0); setDraft(null); }} options={names.map((n) => ({ value: n, label: n }))} />
          <NumberField label={isVec ? 'element' : 'row (output)'} value={rr} min={0} max={rows - 1} step={1} onChange={(v) => { setR(Math.round(v)); setDraft(null); }} width={90} />
          {!isVec && <NumberField label="column (input)" value={cc} min={0} max={cols - 1} step={1} onChange={(v) => { setC(Math.round(v)); setDraft(null); }} width={90} />}
        </div>
        <p>Shape {shapeStr(isVec ? [rows] : [rows, cols])}. Current value: <strong className="mono">{cur.toFixed(6)}</strong>. Selecting a coordinate is inspection only and changes nothing.</p>
        <div className="row">
          <NumberField label="proposed new value" value={draft ?? cur} step={0.1} onChange={setDraft} width={130} />
          <button onClick={() => setDraft(cur * -1)}>flip sign</button>
          <button onClick={() => setDraft(0)}>set 0</button>
        </div>
        {eff && (
          <div className="banner-edit" style={{ marginTop: 10 }}>
            <div>Downstream effect of this one change on prompt “{state.tokens.map(tok).join(' ')}” (computed, not yet applied):</div>
            <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
              {eff.perLayer.map((d, l) => <li key={l}>block {l} output: max change {d === 0 ? <strong>0 (unaffected)</strong> : d.toExponential(3)}</li>)}
              <li>logits: max change {eff.logitDiff.toExponential(3)}</li>
            </ul>
            {eff.layer !== null && eff.layer > 0 && <div className="faint">Blocks before {eff.layer} are unaffected: a block's weights only influence later computation.</div>}
            <button className="small" onClick={() => { dispatch({ type: 'edit', name: tname, row: rr, col: isVec ? null : cc, value: draft! }); setDraft(null); }}>Apply this edit to the shared toy simulator</button>
          </div>
        )}
        <p className="faint" style={{ fontSize: '0.85rem' }}>Applying marks the simulator as edited and invalidates its K/V cache; “restore fixture” returns the exact seeded weights.</p>
      </Section>
    </>
  );
}

export function UnitLab() {
  const { profile, weights, trace, state } = useToy();
  const [layer, setLayer] = useState(0);
  const [pos, setPos] = useState(state.tokens.length - 1);
  const [ch, setCh] = useState(3);
  const [out, setOut] = useState(0);
  const [expert, setExpert] = useState(0);
  const P = trace[Math.min(pos, trace.length - 1)];
  const L = P.layers[layer];
  const fw = profile.kind === 'dense' ? weights.layers[layer].ffn! : weights.layers[layer].experts![expert];
  const width = fw.gate.length;
  const j = Math.min(ch, width - 1);
  const x = L.ln2;
  const gp = x.map((v, i) => v * fw.gate[j][i]);
  const up = x.map((v, i) => v * fw.up[j][i]);
  const a = gp.reduce((s, v) => s + v, 0), u = up.reduce((s, v) => s + v, 0);
  const z = silu(a) * u;
  const zAll = fw.gate.map((g, k) => silu(dot(x, g)) * dot(x, fw.up[k]));
  const yo = fw.down[out].reduce((s, w, k) => s + w * zAll[k], 0);
  const selected = profile.kind === 'moe' ? L.moe!.route.selected.includes(expert) : true;
  return (
    <Section title="2 · Unit microscope: one SwiGLU gated channel" badges={<Evidence kind="simulation" />}>
      <ToyControls />
      <PromptEditor />
      <div className="row">
        <NumberField label="block" value={layer} min={0} max={profile.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={80} />
        <NumberField label="token position" value={Math.min(pos, state.tokens.length - 1)} min={0} max={state.tokens.length - 1} step={1} onChange={(v) => setPos(Math.round(v))} width={80} />
        {profile.moe && <NumberField label="expert" value={expert} min={0} max={profile.moe.numExperts - 1} step={1} onChange={(v) => setExpert(Math.round(v))} width={80} />}
        <NumberField label={`channel j (0–${width - 1})`} value={j} min={0} max={width - 1} step={1} onChange={(v) => setCh(Math.round(v))} width={90} />
        <NumberField label={`output feature (0–${profile.d - 1})`} value={out} min={0} max={profile.d - 1} step={1} onChange={(v) => setOut(Math.round(v))} width={90} />
      </div>
      {profile.moe && !selected && <div className="callout warn">Expert {expert} was <strong>not selected</strong> by the router for this token in block {layer} (selected: {L.moe!.route.selected.join(', ')}). The numbers below show what it would compute; in the real forward pass it is skipped for this token.</div>}
      <div className="grid2" style={{ marginTop: 8 }}>
        <div>
          <VectorView label={`input x = RMSNorm(h) at position ${P.pos} (“${tok(P.token)}”), block ${layer} — activations`} values={x} />
          <VectorView label={`row ${j} of W_gate (incoming weights)`} values={fw.gate[j]} kind="par" />
          <VectorView label="products x_i · W_gate[j, i]" values={gp} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>gate pre-activation a = Σ = {fmtNum(a)}<br />SiLU(a) = a·σ(a) = {fmtNum(a)} × {fmtNum(sigmoid(a))} = <strong>{fmtNum(silu(a))}</strong></p>
          <VectorView label={`row ${j} of W_up (incoming weights)`} values={fw.up[j]} kind="par" />
          <VectorView label="products x_i · W_up[j, i]" values={up} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>up value u = Σ = {fmtNum(u)}</p>
          <p className="mono" style={{ fontSize: '0.95rem' }}>gated channel z_{j} = SiLU(a) × u = <strong>{fmtNum(z)}</strong></p>
        </div>
        <div>
          <VectorView label={`column ${j} of W_down (outgoing weights to all ${profile.d} features)`} values={fw.down.map((r) => r[j])} kind="par" highlight={out} />
          <VectorView label={`all ${width} gated channels z (activations)`} values={zAll} highlight={j} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>
            output feature {out} = Σ_k z_k · W_down[{out}, k] = {fmtNum(yo)}<br />
            contribution of channel {j}: z_{j} · W_down[{out}, {j}] = {fmtNum(z)} × {fmtNum(fw.down[out][j])} = <strong>{fmtNum(z * fw.down[out][j])}</strong>
          </p>
          <div className="callout">
            A gated channel uses <strong>two</strong> weighted sums, a nonlinearity and a multiplication, then is mixed back by a column of W_down. In Qwen3-8B the same structure has d = {fmtInt(QWEN3_8B.d)} inputs and f = {fmtInt(QWEN3_8B.ffnWidth!)} channels; each Qwen3-30B-A3B expert has {QWEN3_30B_A3B.moe!.expertWidth} channels.
          </div>
        </div>
      </div>
    </Section>
  );
}

export function MatrixLab() {
  const samples = useSamples();
  const [m, setM] = useState<'qwen3-8b' | 'qwen3-30b-a3b' | 'toy-dense' | 'toy-moe'>('qwen3-8b');
  const prof = m === 'qwen3-8b' ? QWEN3_8B : m === 'qwen3-30b-a3b' ? QWEN3_30B_A3B : null;
  return (
    <>
      <Section title="3 · Matrix explorer" badges={<><Evidence kind="verified" /><Evidence kind="sample" /></>}>
        <p>Every learned tensor family, with exact logical shape, dtype and parameter count. Opening one shows a bounded 16 × 16 window with exact indices, an extent indicator, the checkpoint tensor and its byte location, and real values where a checked sample exists.</p>
        <Select label="Model" value={m} onChange={setM} options={[{ value: 'qwen3-8b', label: 'Qwen3-8B' }, { value: 'qwen3-30b-a3b', label: 'Qwen3-30B-A3B' }, { value: 'toy-dense', label: 'Toy simulator (dense)' }, { value: 'toy-moe', label: 'Toy simulator (MoE)' }]} />
        <div className="table-wrap" style={{ marginTop: 8 }}>
          <table className="t">
            <thead><tr><th>Tensor family</th><th>Checkpoint name</th><th>Shape</th><th className="num">per tensor</th><th className="num">count</th><th /></tr></thead>
            <tbody>
              {tensorFamilies(prof ?? (m === 'toy-dense' ? TOY_DENSE : TOY_MOE)).map((f) => (
                <tr key={f.key}>
                  <td>{f.label}</td>
                  <td className="mono" style={{ fontSize: '0.8rem' }}>{f.pathTemplate}</td>
                  <td className="mono">{shapeStr(f.shape)}</td>
                  <td className="num">{fmtInt(f.shape.reduce((a, b) => a * b, 1))}</td>
                  <td className="num">{fmtInt(f.multiplicity)}</td>
                  <td><a href={explorerHref({ m, lv: 2, t: f.key, b: f.scope === 'model' ? undefined : 0, e: f.scope === 'expert' ? 0 : undefined })}>open</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Actual checkpoint samples shipped with this site" badges={<Evidence kind="sample" />}>
        {samples.status === 'loading' && <p className="muted">Loading sample manifest…</p>}
        {samples.status === 'error' && <div className="callout warn">Real samples could not be loaded ({samples.error}). The structural catalog and the toy microscope remain available.</div>}
        {samples.status === 'ok' && (
          <>
            <p className="muted">Read with HTTP Range requests from the pinned shards; {fmtInt(samples.data.bytesUsed)} bytes transferred in total under a {fmtInt(samples.data.budgetBytes)}-byte budget (headers included). Each is a tiny tile, not a model and not an inference trace.</p>
            <div className="table-wrap"><table className="t">
              <thead><tr><th>Model @ revision</th><th>Tensor</th><th>Indices</th><th>dtype</th><th>SHA-256 of raw bytes</th><th /></tr></thead>
              <tbody>
                {samples.data.samples.map((s) => {
                  const p = s.modelKey === 'qwen3-8b' ? QWEN3_8B : QWEN3_30B_A3B;
                  const loc = locateTensor(p, s.tensor);
                  return (
                    <tr key={s.id}>
                      <td>{s.model} @ <span className="mono">{s.revision.slice(0, 10)}</span></td>
                      <td className="mono" style={{ fontSize: '0.8rem' }}>{s.tensor}<br /><span className="faint">{s.shard}</span></td>
                      <td className="mono">{s.index.rows ? `r ${s.index.rows[0]}–${s.index.rows[1]}, ` : ''}c {s.index.cols[0]}–{s.index.cols[1]}</td>
                      <td>{s.dtype}</td>
                      <td className="mono" style={{ fontSize: '0.75rem' }}>{s.rawSha256.slice(0, 20)}…</td>
                      <td>{loc && <a href={explorerHref({ m: s.modelKey, lv: 2, t: loc.key, b: loc.b, e: loc.e })}>view</a>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table></div>
          </>
        )}
      </Section>
    </>
  );
}

