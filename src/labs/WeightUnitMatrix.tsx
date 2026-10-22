import { useMemo, useState } from 'react';
import { useToy } from '../state/toyStore';
import { forwardFull, toyTensor, toyTensorNames, withEditedWeight } from '../math/toy';
import { maxAbsDiff, dot } from '../math/linalg';
import { silu, sigmoid } from '../math/ops';
import { Evidence, NumberField, Section, Select, VectorView } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { TinyExample } from './TinyExample';
import { QWEN3_30B_A3B, QWEN3_8B, TOY_DENSE, TOY_MOE } from '../models/profiles';
import { tensorFamilies } from '../models/shapes';
import { explorerHref } from '../state/route';
import { useSamples } from '../data/data';
import { locateTensor } from '../explorer/locate';
import { useLang } from '../i18n/i18n';

function useT() {
  const l = useLang();
  return { ...l, T: (en: string, sk: string) => (l.lang === 'sk' ? sk : en) };
}

export function WeightLab() {
  const { profile, weights, state, trace, dispatch } = useToy();
  const { T, fnum, shape } = useT();
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
      <Section title={T('1 · The tiny example', '1 · Malý príklad')} badges={<Evidence kind="simulation" />}>
        <TinyExample />
      </Section>
      <Section title={T('Weight microscope: change one stored number in the toy model', 'Mikroskop váhy: zmeňte jedno uložené číslo v hračkovom modeli')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <div className="row">
          <Select label={T('Toy tensor (Qwen-style name)', 'Hračkový tenzor (názov ako v Qwen)')} value={tname} onChange={(v) => { setName(v); setR(0); setC(0); setDraft(null); }} options={names.map((n) => ({ value: n, label: n }))} />
          <NumberField label={isVec ? T('element', 'prvok') : T('row (output)', 'riadok (výstup)')} value={rr} min={0} max={rows - 1} step={1} onChange={(v) => { setR(Math.round(v)); setDraft(null); }} width={90} />
          {!isVec && <NumberField label={T('column (input)', 'stĺpec (vstup)')} value={cc} min={0} max={cols - 1} step={1} onChange={(v) => { setC(Math.round(v)); setDraft(null); }} width={90} />}
        </div>
        <p>{T('Shape', 'Tvar')} {shape(isVec ? [rows] : [rows, cols])}. {T('Current value', 'Aktuálna hodnota')}: <strong className="mono">{fnum(cur, 6)}</strong>. {T('Selecting a coordinate is inspection only and changes nothing.', 'Výber súradnice je len prezeranie a nič nemení.')}</p>
        <div className="row">
          <NumberField label={T('proposed new value', 'navrhovaná nová hodnota')} value={draft ?? cur} step={0.1} onChange={setDraft} width={130} />
          <button onClick={() => setDraft(cur * -1)}>{T('flip sign', 'obrátiť znamienko')}</button>
          <button onClick={() => setDraft(0)}>{T('set 0', 'nastaviť 0')}</button>
        </div>
        {eff && (
          <div className="banner-edit" style={{ marginTop: 10 }}>
            <div>{T(`Downstream effect of this one change on prompt “${state.tokens.map(tok).join(' ')}” (computed, not yet applied):`, `Účinok tejto jednej zmeny na zadanie „${state.tokens.map(tok).join(' ')}“ (vypočítaný, zatiaľ nepoužitý):`)}</div>
            <ul style={{ margin: '4px 0', paddingLeft: 18 }}>
              {eff.perLayer.map((d, l) => <li key={l}>{T(`block ${l} output: max change`, `výstup bloku ${l}: najväčšia zmena`)} {d === 0 ? <strong>{T('0 (unaffected)', '0 (bez vplyvu)')}</strong> : fnum(d, 4)}</li>)}
              <li>{T('logits: max change', 'logity: najväčšia zmena')} {fnum(eff.logitDiff, 4)}</li>
            </ul>
            {eff.layer !== null && eff.layer > 0 && <div className="faint">{T(`Blocks before ${eff.layer} are unaffected: a block's weights only influence later computation.`, `Bloky pred blokom ${eff.layer} nie sú ovplyvnené: váhy bloku ovplyvňujú len neskorší výpočet.`)}</div>}
            <button className="small" onClick={() => { dispatch({ type: 'edit', name: tname, row: rr, col: isVec ? null : cc, value: draft! }); setDraft(null); }}>{T('Apply this edit to the shared toy simulator', 'Použiť túto úpravu v spoločnom hračkovom simulátore')}</button>
          </div>
        )}
        <p className="faint" style={{ fontSize: '0.85rem' }}>{T('Applying marks the simulator as edited and invalidates its K/V cache; “restore fixture” returns the exact seeded weights.', 'Použitie označí simulátor ako upravený a zneplatní jeho K/V cache; „obnoviť pôvodné“ vráti presné váhy z osiva.')}</p>
      </Section>
    </>
  );
}

export function UnitLab() {
  const { profile, weights, trace, state } = useToy();
  const { T, fnum, num } = useT();
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
    <Section title={T('2 · Unit microscope: one SwiGLU gated channel', '2 · Mikroskop jednotky: jeden hradlovaný kanál SwiGLU')} badges={<Evidence kind="simulation" />}>
      <ToyControls />
      <PromptEditor />
      <div className="row">
        <NumberField label={T('block', 'blok')} value={layer} min={0} max={profile.layers - 1} step={1} onChange={(v) => setLayer(Math.round(v))} width={80} />
        <NumberField label={T('token position', 'pozícia tokenu')} value={Math.min(pos, state.tokens.length - 1)} min={0} max={state.tokens.length - 1} step={1} onChange={(v) => setPos(Math.round(v))} width={80} />
        {profile.moe && <NumberField label={T('expert', 'expert')} value={expert} min={0} max={profile.moe.numExperts - 1} step={1} onChange={(v) => setExpert(Math.round(v))} width={80} />}
        <NumberField label={T(`channel j (0–${width - 1})`, `kanál j (0–${width - 1})`)} value={j} min={0} max={width - 1} step={1} onChange={(v) => setCh(Math.round(v))} width={90} />
        <NumberField label={T(`output feature (0–${profile.d - 1})`, `výstupná vlastnosť (0–${profile.d - 1})`)} value={out} min={0} max={profile.d - 1} step={1} onChange={(v) => setOut(Math.round(v))} width={90} />
      </div>
      {profile.moe && !selected && <div className="callout warn">{T(`Expert ${expert} was not selected by the router for this token in block ${layer} (selected: ${L.moe!.route.selected.join(', ')}). The numbers below show what it would compute; in the real forward pass it is skipped for this token.`, `Experta ${expert} smerovač pre tento token v bloku ${layer} nevybral (vybraní: ${L.moe!.route.selected.join(', ')}). Čísla nižšie ukazujú, čo by vypočítal; v skutočnom priechode sa pre tento token preskočí.`)}</div>}
      <div className="grid2" style={{ marginTop: 8 }}>
        <div>
          <VectorView label={T(`input x = RMSNorm(h) at position ${P.pos} (“${tok(P.token)}”), block ${layer} — activations`, `vstup x = RMSNorm(h) na pozícii ${P.pos} („${tok(P.token)}“), blok ${layer} — aktivácie`)} values={x} />
          <VectorView label={T(`row ${j} of W_gate (incoming weights)`, `riadok ${j} matice W_gate (vstupné váhy)`)} values={fw.gate[j]} kind="par" />
          <VectorView label={T('products x_i · W_gate[j, i]', 'súčiny x_i · W_gate[j, i]')} values={gp} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>{T('gate pre-activation', 'pred-aktivácia hradla')} a = Σ = {fnum(a)}<br />SiLU(a) = a·σ(a) = {fnum(a)} × {fnum(sigmoid(a))} = <strong>{fnum(silu(a))}</strong></p>
          <VectorView label={T(`row ${j} of W_up (incoming weights)`, `riadok ${j} matice W_up (vstupné váhy)`)} values={fw.up[j]} kind="par" />
          <VectorView label={T('products x_i · W_up[j, i]', 'súčiny x_i · W_up[j, i]')} values={up} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>{T('up value', 'hodnota up')} u = Σ = {fnum(u)}</p>
          <p className="mono" style={{ fontSize: '0.95rem' }}>{T('gated channel', 'hradlovaný kanál')} z_{j} = SiLU(a) × u = <strong>{fnum(z)}</strong></p>
        </div>
        <div>
          <VectorView label={T(`column ${j} of W_down (outgoing weights to all ${profile.d} features)`, `stĺpec ${j} matice W_down (výstupné váhy do všetkých ${profile.d} vlastností)`)} values={fw.down.map((r) => r[j])} kind="par" highlight={out} />
          <VectorView label={T(`all ${width} gated channels z (activations)`, `všetkých ${width} hradlovaných kanálov z (aktivácie)`)} values={zAll} highlight={j} />
          <p className="mono" style={{ fontSize: '0.88rem' }}>
            {T('output feature', 'výstupná vlastnosť')} {out} = Σ_k z_k · W_down[{out}, k] = {fnum(yo)}<br />
            {T(`contribution of channel ${j}`, `príspevok kanála ${j}`)}: z_{j} · W_down[{out}, {j}] = {fnum(z)} × {fnum(fw.down[out][j])} = <strong>{fnum(z * fw.down[out][j])}</strong>
          </p>
          <div className="callout">
            {T(`A gated channel uses two weighted sums, a nonlinearity and a multiplication, then is mixed back by a column of W_down. In Qwen3-8B the same structure has d = ${num(QWEN3_8B.d)} inputs and f = ${num(QWEN3_8B.ffnWidth!)} channels; each Qwen3-30B-A3B expert has ${QWEN3_30B_A3B.moe!.expertWidth} channels.`, `Hradlovaný kanál používa dva vážené súčty, nelinearitu a násobenie a potom ho stĺpec matice W_down zmieša späť. V Qwen3-8B má rovnaká štruktúra d = ${num(QWEN3_8B.d)} vstupov a f = ${num(QWEN3_8B.ffnWidth!)} kanálov; každý expert Qwen3-30B-A3B má ${QWEN3_30B_A3B.moe!.expertWidth} kanálov.`)}
          </div>
        </div>
      </div>
    </Section>
  );
}

export function MatrixLab() {
  const samples = useSamples();
  const { T, num, shape, lang } = useT();
  const [m, setM] = useState<'qwen3-8b' | 'qwen3-30b-a3b' | 'toy-dense' | 'toy-moe'>('qwen3-8b');
  const prof = m === 'qwen3-8b' ? QWEN3_8B : m === 'qwen3-30b-a3b' ? QWEN3_30B_A3B : null;
  return (
    <>
      <Section title={T('3 · Matrix explorer', '3 · Prieskumník matíc')} badges={<><Evidence kind="verified" /><Evidence kind="sample" /></>}>
        <p>{T('Every learned tensor family, with exact logical shape, dtype and parameter count. Opening one shows a bounded 16 × 16 window with exact indices, an extent indicator, the checkpoint tensor and its byte location, and real values where a checked sample exists.', 'Každá rodina naučených tenzorov s presným logickým tvarom, typom čísel a počtom parametrov. Po otvorení sa zobrazí obmedzený výrez 16 × 16 s presnými indexmi, ukazovateľom rozsahu, tenzorom v checkpointe a jeho umiestnením v bajtoch a skutočnými hodnotami tam, kde existuje overená vzorka.')}</p>
        <Select label={T('Model', 'Model')} value={m} onChange={setM} options={[{ value: 'qwen3-8b', label: 'Qwen3-8B' }, { value: 'qwen3-30b-a3b', label: 'Qwen3-30B-A3B' }, { value: 'toy-dense', label: T('Toy simulator (dense)', 'Hračkový simulátor (hustý)') }, { value: 'toy-moe', label: T('Toy simulator (MoE)', 'Hračkový simulátor (MoE)') }]} />
        <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}>
          <table className="t">
            <thead><tr><th>{T('Tensor family', 'Rodina tenzorov')}</th><th>{T('Checkpoint name', 'Názov v checkpointe')}</th><th>{T('Shape', 'Tvar')}</th><th className="num">{T('per tensor', 'na tenzor')}</th><th className="num">{T('count', 'počet')}</th><th /></tr></thead>
            <tbody>
              {tensorFamilies(prof ?? (m === 'toy-dense' ? TOY_DENSE : TOY_MOE)).map((f) => (
                <tr key={f.key}>
                  <td>{lang === 'sk' ? f.labelSk : f.label}</td>
                  <td className="mono" style={{ fontSize: '0.8rem' }}>{f.pathTemplate}</td>
                  <td className="mono">{shape(f.shape)}</td>
                  <td className="num">{num(f.shape.reduce((a, b) => a * b, 1))}</td>
                  <td className="num">{num(f.multiplicity)}</td>
                  <td><a href={explorerHref({ m, lv: 2, t: f.key, b: f.scope === 'model' ? undefined : 0, e: f.scope === 'expert' ? 0 : undefined })}>{T('open', 'otvoriť')}</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title={T('Actual checkpoint samples shipped with this site', 'Skutočné vzorky z checkpointov dodané s touto stránkou')} badges={<Evidence kind="sample" />}>
        {samples.status === 'loading' && <p className="muted">{T('Loading sample manifest…', 'Načítavam zoznam vzoriek…')}</p>}
        {samples.status === 'error' && <div className="callout warn">{T(`Real samples could not be loaded (${samples.error}). The structural catalog and the toy microscope remain available.`, `Skutočné vzorky sa nepodarilo načítať (${samples.error}). Štrukturálny katalóg a hračkový mikroskop zostávajú dostupné.`)}</div>}
        {samples.status === 'ok' && (
          <>
            <p className="muted">{T(`Read with HTTP Range requests from the pinned shards; ${num(samples.data.bytesUsed)} bytes transferred in total under a ${num(samples.data.budgetBytes)}-byte budget (headers included). Each is a tiny tile, not a model and not an inference trace.`, `Prečítané požiadavkami HTTP Range zo súborov v pevnej verzii; spolu ${num(samples.data.bytesUsed)} prenesených bajtov pri limite ${num(samples.data.budgetBytes)} bajtov (vrátane hlavičiek). Každá vzorka je malý výrez, nie model ani záznam behu.`)}</p>
            <div className="table-wrap" tabIndex={0}><table className="t">
              <thead><tr><th>{T('Model @ revision', 'Model @ verzia')}</th><th>{T('Tensor', 'Tenzor')}</th><th>{T('Indices', 'Indexy')}</th><th>dtype</th><th>{T('SHA-256 of raw bytes', 'SHA-256 surových bajtov')}</th><th /></tr></thead>
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
                      <td>{loc && <a href={explorerHref({ m: s.modelKey, lv: 2, t: loc.key, b: loc.b, e: loc.e })}>{T('view', 'zobraziť')}</a>}</td>
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
