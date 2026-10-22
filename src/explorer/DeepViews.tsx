import { useMemo, useState } from 'react';
import type { CoreProfile } from '../models/profiles';
import { packedExpertMapping, prod, tensorName, type TensorFamily } from '../models/shapes';
import { locateTensor } from './locate';
export { locateTensor };
import { Evidence, ExportButton, NumberField, SimBanner, Tex, VectorView } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { fileOffset, sampleFor, useCatalog, useOffsets, useSamples } from '../data/data';
import { useToy } from '../state/toyStore';
import { explorerHref, type ExplorerState } from '../state/route';
import { toyCounterpart, toyTrace, toyValue, unitInput } from './toyCounterpart';
import { bf16Parts, toBf16Bits, isExactBf16 } from '../math/bf16';
import { forwardFull, toyTensor, withEditedWeight, TOY_VOCAB } from '../math/toy';
import { silu, sigmoid } from '../math/ops';
import { dot, maxAbsDiff } from '../math/linalg';
import { kvBytesPerToken, fmtBytes } from '../math/memory';
import { accounting } from '../models/shapes';

const TILE = 16;



interface DeepProps { p: CoreProfile; f: TensorFamily; s: ExplorerState; go: (s: Partial<ExplorerState>) => void }

function useRealValues(p: CoreProfile, name: string) {
  const samples = useSamples();
  const list = samples.status === 'ok' ? samples.data.samples : undefined;
  return { samples, list, has: !!list?.some((x) => x.tensor === name && x.modelKey === p.id) };
}

/** LEVEL 2 — matrix / vector tile with exact indices and an extent indicator. Never renders one element per real parameter. */
export function MatrixView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const { lang, num: N, fnum, shape } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const flabel = lang === 'sk' ? f.labelSk : f.label;
  const rows = f.shape[0], cols = f.shape.length > 1 ? f.shape[1] : 1;
  const isVec = f.shape.length === 1;
  const name = tensorName(f, s.b ?? 0, s.e ?? 0);
  const selR = s.r ?? 0, selC = s.c ?? 0;
  const [r0, setR0] = useState(() => Math.floor(selR / TILE) * TILE);
  const [c0, setC0] = useState(() => Math.floor((isVec ? selR : selC) / TILE) * TILE);
  const real = useRealValues(p, name);
  const isToy = p.status === 'toy';
  const tt = useMemo(() => (isToy ? toyTrace(p, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [isToy, p, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]);
  const catalog = useCatalog(p.status === 'official' ? p.id : null);
  const offsets = useOffsets(p.status === 'official' ? p.id : null);
  const value = (r: number, c: number): { v: number | null; kind: 'toy' | 'sample' | 'none' } => {
    if (isToy && tt) {
      const t = toyTensor(tt.w, name);
      return { v: isVec ? (t as number[])[c] : (t as number[][])[r][c], kind: 'toy' };
    }
    const hit = sampleFor(real.list?.filter((x) => x.modelKey === p.id), name, isVec ? c : r, isVec ? undefined : c);
    return hit && hit.value !== null ? { v: hit.value, kind: 'sample' } : { v: null, kind: 'none' };
  };
  const nr = isVec ? 1 : Math.min(TILE, rows - r0), nc = Math.min(TILE, cols - c0);
  const fo = offsets.status === 'ok' ? fileOffset(offsets.data, name, isVec ? c0 : r0, isVec ? null : c0) : null;
  const mapping = f.scope === 'expert' ? packedExpertMapping(p, s.e ?? 0) : null;
  const shard = catalog.status === 'ok' && f.scope !== 'model' ? catalog.data.layerShards[String(s.b ?? 0)] : null;
  const anyReal = !isToy && Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => value(r0 + i, c0 + j).kind === 'sample')).flat().some(Boolean);
  const mm = 140;
  const bw = Math.max(3, (nc / cols) * mm), bh = isVec ? 10 : Math.max(3, (nr / rows) * mm);
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>{flabel} <span className="faint mono" style={{ fontSize: '0.85rem' }}>{name}</span></h2>
        <div className="badge-row">
          <Evidence kind={isToy ? 'simulation' : 'verified'} title="Shape, dtype and name" />
          {anyReal && <Evidence kind="sample" />}
          {!isToy && !anyReal && (real.samples.status === 'loading' ? <span className="coverage">{T('checking real samples…', 'kontrolujem skutočné vzorky…')}</span> : <Evidence kind="unknown" title="Numerical values in this window are not loaded" />)}
        </div>
      </div>
      {isToy && <SimBanner />}
      <div className="grid2" style={{ marginTop: 10 }}>
        <div className="card">
          <dl className="kv">
            <dt>{T('Logical shape', 'Logický tvar')}</dt><dd className="mono">{shape(f.shape)} — {f.axes.join(' × ')}</dd>
            <dt>{T('Parameters', 'Parametre')}</dt><dd className="mono">{f.shape.map((v) => N(v)).join(' × ')} = {N(prod(f.shape))}</dd>
            <dt>{T('Storage dtype', 'Typ uložených čísel')}</dt><dd>{p.status === 'official' ? T(`${p.dtype} (2 bytes per value) in the checkpoint`, `${p.dtype} (2 bajty na hodnotu) v checkpointe`) : T('float64 in the browser simulator', 'float64 v simulátore v prehliadači')}</dd>
            <dt>{T('Orientation', 'Orientácia')}</dt><dd>{isVec ? T('vector: one gain per feature', 'vektor: jeden zisk pre každú vlastnosť') : T('PyTorch [out, in]: row = output coordinate, column = input feature; Y = X Wᵀ', 'PyTorch [výstup, vstup]: riadok = výstupná súradnica, stĺpec = vstupná vlastnosť; Y = X Wᵀ')}</dd>
            {p.status === 'official' && <><dt>Checkpoint</dt><dd className="mono">{fo ? `${fo.shard} · ${T('bytes', 'bajty')} ${N(fo.byte)}…` : shard ? `shard(s) ${shard.join(', ')}` : offsets.status === 'loading' ? T('loading byte map…', 'načítavam mapu bajtov…') : T('see catalog', 'pozri katalóg')}</dd></>}
            <dt>{T('Implementation', 'Implementácia')}</dt><dd className="mono">{f.symbol}</dd>
          </dl>
          {!isVec && (
            <p className="muted" style={{ fontSize: '0.88rem' }}>
              {T(`Batched use: for input X of shape [B, T, ${N(f.shape[1])}] the same matrix is applied at every one of the B·T token positions, producing [B, T, ${N(f.shape[0])}]. The weights are stored once; the products are activations.`, `Použitie v dávke: pre vstup X tvaru [B, T, ${N(f.shape[1])}] sa tá istá matica použije na každú z B·T pozícií tokenov a vznikne [B, T, ${N(f.shape[0])}]. Váhy sú uložené raz; súčiny sú aktivácie.`)}
            </p>
          )}
          {mapping && (
            <details>
              <summary>{T('Logical expert matrices ↔ physical tensors', 'Logické matice experta ↔ fyzické tenzory')}</summary>
              <table className="t"><tbody>
                <tr><th>{T('Logical', 'Logická')}</th><th>{T('Checkpoint file tensor', 'Tenzor v súbore checkpointu')}</th><th>{T('Runtime (Transformers, packed)', 'Počas behu (Transformers, zbalené)')}</th></tr>
                {(['gate', 'up', 'down'] as const).map((k) => (
                  <tr key={k}><td>{k}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{mapping.checkpoint[k].replace('{layer}', String(s.b ?? 0))}</td><td className="mono" style={{ fontSize: '0.78rem' }}>{mapping.runtime[k]}</td></tr>
                ))}
              </tbody></table>
              <p className="faint" style={{ fontSize: '0.82rem' }}>{T(mapping.note, 'Zbalené tenzory počas behu sú len iným usporiadaním tých istých čísel (MergeModulelist cez expertov, potom spojenie gate a up v dimenzii 1). V súčtoch parametrov sa čísla počítajú raz.')}</p>
            </details>
          )}
        </div>
        <div className="card">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong>{T('Tile window', 'Výrez matice')}</strong>
            <span className="faint" style={{ fontSize: '0.82rem' }}>{T('showing', 'zobrazené')} {isVec ? T(`elements ${c0}–${c0 + nc - 1}`, `prvky ${c0}–${c0 + nc - 1}`) : T(`rows ${r0}–${r0 + nr - 1}, cols ${c0}–${c0 + nc - 1}`, `riadky ${r0}–${r0 + nr - 1}, stĺpce ${c0}–${c0 + nc - 1}`)} {T('of', 'z')} {shape(f.shape)}</span>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            {!isVec && <NumberField label={T('first row', 'prvý riadok')} value={r0} min={0} max={Math.max(0, rows - 1)} step={TILE} onChange={(v) => setR0(Math.min(Math.floor(v / TILE) * TILE, Math.max(0, rows - 1)))} width={100} />}
            <NumberField label={isVec ? T('first element', 'prvý prvok') : T('first column', 'prvý stĺpec')} value={c0} min={0} max={Math.max(0, cols - 1)} step={TILE} onChange={(v) => setC0(Math.min(Math.floor(v / TILE) * TILE, Math.max(0, cols - 1)))} width={100} />
            <svg width={mm + 4} height={(isVec ? 12 : mm) + 4} role="img" aria-label={T(`Extent indicator: the window covers ${nr * nc} of ${N(prod(f.shape))} values`, `Ukazovateľ rozsahu: výrez pokrýva ${nr * nc} z ${N(prod(f.shape))} hodnôt`)}>
              <rect x={2} y={2} width={mm} height={isVec ? 10 : mm} fill="var(--bg-2)" stroke="var(--line)" />
              <rect x={2 + (c0 / cols) * mm} y={2 + (isVec ? 0 : (r0 / rows) * mm)} width={bw} height={bh} fill="var(--focus)" />
            </svg>
          </div>
          <p className="faint" style={{ fontSize: '0.8rem', margin: '4px 0' }}>{T(`Rendering limit: at most ${TILE}×${TILE} cells at a time (no element is created per real parameter).`, `Obmedzenie vykresľovania: najviac ${TILE}×${TILE} buniek naraz (pre skutočné parametre sa nevytvára prvok na každý).`)}</p>
        </div>
      </div>
      <div className="card" style={{ marginTop: 12, overflowX: 'auto' }}>
        <div className="matrix-grid" style={{ gridTemplateColumns: `52px repeat(${nc}, minmax(34px, 1fr))` }} role="grid" aria-label={T(`${f.label} values, window`, `${flabel}: hodnoty, výrez`)}>
          <div role="row" style={{ display: 'contents' }}>
            <div className="cell hdr" role="columnheader">{isVec ? 'i' : 'r \\ c'}</div>
            {Array.from({ length: nc }, (_, j) => <div key={j} className="cell hdr" role="columnheader">{c0 + j}</div>)}
          </div>
          {Array.from({ length: nr }, (_, i) => {
            const r = isVec ? 0 : r0 + i;
            return <div key={`row${i}`} role="row" style={{ display: 'contents' }}>{[
              <div key={`h${i}`} className="cell hdr" role="rowheader">{isVec ? '' : r}</div>,
              ...Array.from({ length: nc }, (_, j) => {
                const c = c0 + j;
                const v = value(r, c);
                const isSel = isVec ? s.r === c : s.r === r && s.c === c;
                const isRow = !isVec && s.r === r;
                return (
                  <button key={`${i}-${j}`} role="gridcell" className={`cell${v.kind === 'none' ? ' na' : ''}${isSel ? ' sel' : ''}${isRow && !isSel ? ' hl' : ''}`}
                    aria-label={lang === 'sk' ? `${isVec ? `prvok ${c}` : `riadok ${r}, stĺpec ${c}`}: ${v.v === null ? 'hodnota nenačítaná' : v.v}` : `${isVec ? `element ${c}` : `row ${r}, column ${c}`}: ${v.v === null ? 'value not loaded' : v.v}`}
                    title={v.v === null ? T('Actual value not loaded (outside the bounded real sample)', 'Skutočná hodnota nie je načítaná (mimo obmedzenej skutočnej vzorky)') : `${v.kind === 'sample' ? T('Real checkpoint value', 'Skutočná hodnota z checkpointu') : T('Toy value', 'Hračková hodnota')}: ${v.v}`}
                    onClick={() => go(isVec ? { lv: 0, r: c, c: undefined } : { lv: 0, r, c })}>
                    {v.v === null ? '·' : fnum(v.v, 2)}
                  </button>
                );
              }),
            ]}</div>;
          })}
        </div>
        <p className="faint" style={{ fontSize: '0.82rem' }}>
          {T('Click a cell to open that single stored number (Level 0).', 'Kliknutím na bunku otvoríte toto jedno uložené číslo (úroveň 0).')} {!isVec && T('Use “Open row as unit” to see how one output coordinate is computed (Level 1).', 'Tlačidlom „Otvoriť riadok ako jednotku“ uvidíte, ako sa počíta jedna výstupná súradnica (úroveň 1).')}
          {!isToy && !anyReal && T(' Hatched cells: the actual checkpoint values here were not loaded. Only small, provenance-checked tiles are shipped (see Sources). The toy microscope shows the same computation with simulated values.', ' Šrafované bunky: skutočné hodnoty z checkpointu tu nie sú načítané. Dodávajú sa len malé výrezy s overeným pôvodom (pozri Zdroje). Hračkový mikroskop ukazuje ten istý výpočet so simulovanými hodnotami.')}
        </p>
        <div className="row">
          {!isVec && <button onClick={() => go({ lv: 1, r: s.r ?? r0 })}>{T(`Open row ${s.r ?? r0} as unit (Level 1)`, `Otvoriť riadok ${s.r ?? r0} ako jednotku (úroveň 1)`)}</button>}
          <ExportButton filename={`${p.id}-${f.key}-tile.json`} data={() => ({
            model: p.repo ?? p.name, revision: p.revision ?? null, tensor: name, logicalShape: f.shape, dtype: p.status === 'official' ? p.dtype : 'float64 (toy)',
            window: { rows: isVec ? null : [r0, r0 + nr - 1], cols: [c0, c0 + nc - 1] },
            evidence: isToy ? 'illustrative-simulation' : anyReal ? 'actual-checkpoint-sample' : 'verified-structure; values not loaded',
            values: Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => value(isVec ? 0 : r0 + i, c0 + j).v)),
          })} />
        </div>
      </div>
    </div>
  );
}

/** LEVEL 1 — one output coordinate (unit / gated channel): every multiplication, the sum, the nonlinearity. */
export function UnitView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const { lang, num: N, fnum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const flabel = lang === 'sk' ? f.labelSk : f.label;
  const r = s.r ?? 0;
  const cp = toyCounterpart(p, f.key, s.b ?? 0, s.e ?? 0, r, 0);
  const tt = useMemo(() => (cp ? toyTrace(cp.toy, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [cp?.toy, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]); // eslint-disable-line react-hooks/exhaustive-deps
  const [pos, setPos] = useState(toyApi.state.tokens.length - 1);
  const inCount = f.shape.length > 1 ? f.shape[1] : 1;
  const ffnLike = ['gate_proj', 'up_proj', 'down_proj', 'expert_gate_proj', 'expert_up_proj', 'expert_down_proj'].includes(f.key);
  const isDown = f.key.endsWith('down_proj');
  if (!cp || !tt) return <div className="page">{T('No unit view for this tensor.', 'Pre tento tenzor nie je pohľad jednotky.')}</div>;
  const inp = unitInput(cp, tt.trace, tt.w, pos, lang);
  const W = toyTensor(tt.w, cp.name) as number[][];
  const row = W[cp.row];
  const products = inp ? inp.x.map((x, i) => x * row[i]) : [];
  const sum = products.reduce((a, b) => a + b, 0);
  // gated channel view for FFN-like tensors
  const ch = cp.row;
  const L = tt.trace[Math.min(pos, tt.trace.length - 1)].layers[cp.layer];
  let gated: { gPre: number; g: number; u: number; z: number; downCol: number[] } | null = null;
  if (ffnLike) {
    const fw = cp.family.scope === 'expert' ? tt.w.layers[cp.layer].experts![cp.expert] : tt.w.layers[cp.layer].ffn!;
    const chan = isDown ? 0 : ch;
    const gPre = dot(L.ln2, fw.gate[chan]);
    const u = dot(L.ln2, fw.up[chan]);
    gated = { gPre, g: silu(gPre), u, z: silu(gPre) * u, downCol: fw.down.map((rr) => rr[chan]) };
  }
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>{isDown || !ffnLike ? T(`Output coordinate ${r} of ${f.label}`, `Výstupná súradnica ${r} tenzora ${flabel}`) : T(`Gated channel ${r}`, `Hradlovaný kanál ${r}`)}</h2>
      <div className="card">
        {lang === 'sk' ? (
          <p>V modeli <strong>{p.name}</strong> je výstup {r} tenzora <code>{tensorName(f, s.b ?? 0, s.e ?? 0)}</code> vážený súčet <strong>{N(inCount)}</strong> vstupných vlastností s použitím riadku {r} ({N(inCount)} naučených váh):</p>
        ) : (
          <p>In <strong>{p.name}</strong>, output {r} of <code>{tensorName(f, s.b ?? 0, s.e ?? 0)}</code> is a weighted sum of <strong>{N(inCount)}</strong> input features using row {r} ({N(inCount)} learned weights):</p>
        )}
        <Tex block>{`y_{${r}} = \\sum_{i=0}^{${inCount - 1}} x_i \\, W[${r}, i]`}</Tex>
        {ffnLike && !isDown && (
          lang === 'sk' ? (
            <p>V SwiGLU je „neurón“ v skutočnosti <strong>hradlovaný kanál</strong>: kanál j používa riadok j matice W<sub>gate</sub> <em>a</em> riadok j matice W<sub>up</sub> (dva vážené súčty), na hradlo použije SiLU, oba výsledky vynásobí a stĺpec j matice W<sub>down</sub> potom prenesie výsledok do každej vlastnosti reziduálneho prúdu. Klasický obrázok neurónu ako krúžku so šípkami je len prirovnanie.</p>
          ) : (
            <p>In SwiGLU a “neuron” is really a <strong>gated channel</strong>: channel j uses row j of W<sub>gate</sub> <em>and</em> row j of W<sub>up</sub> (two weighted sums), applies SiLU to the gate, multiplies the two, and then column j of W<sub>down</sub> carries the result into every residual feature. The classic circle-with-arrows neuron picture is only an analogy for this.</p>
          )
        )}
        <p className="faint" style={{ fontSize: '0.86rem' }}>{T(`Incoming weights: row ${r} (persistent). Pre-activation: the sum (runtime). Activation / gate: computed from it (runtime). Outgoing weights: ${ffnLike && !isDown ? `column ${r} of W_down` : 'the next layer’s matrix'}.`, `Vstupné váhy: riadok ${r} (trvalé). Pred-aktivácia: súčet (počas behu). Aktivácia / hradlo: vypočítané z neho (počas behu). Výstupné váhy: ${ffnLike && !isDown ? `stĺpec ${r} matice W_down` : 'matica ďalšej vrstvy'}.`)}</p>
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <SimBanner>
          {cp.mapped ? (lang === 'sk' ? <>Vypočítané v <strong>{cp.toy.name}</strong> na tenzore s rovnakou úlohou: <code>{cp.name}</code> riadok {cp.row} ({cp.toy.d}-rozmerný namiesto {N(inCount)}). Indexy boli zúžené na menší rozsah hračky.</> : <>Computed in the <strong>{cp.toy.name}</strong> on the same tensor role: <code>{cp.name}</code> row {cp.row} ({cp.toy.d}-dimensional instead of {N(inCount)}). Indices were folded into the toy's smaller extent.</>) : <>{T('Toy simulator values.', 'Hodnoty hračkového simulátora.')}</>}
          {!tt.shared && T(' Uses the unedited seeded toy weights.', ' Používa neupravené hračkové váhy z osiva.')}
        </SimBanner>
        <div className="row" style={{ marginTop: 8 }}>
          <label className="field inline">{T('Token position', 'Pozícia tokenu')}
            <select value={pos} onChange={(e) => setPos(Number(e.target.value))}>
              {toyApi.state.tokens.map((t, i) => <option key={i} value={i}>{i}: {TOY_VOCAB[t]}</option>)}
            </select>
          </label>
          <span className="faint" style={{ fontSize: '0.85rem' }}>{T('Changing the input recomputes activations; the weights stay fixed.', 'Zmena vstupu prepočíta aktivácie; váhy zostávajú rovnaké.')}</span>
        </div>
        {inp ? (
          <>
            <p className="faint" style={{ fontSize: '0.85rem', marginBottom: 2 }}>{T('Input', 'Vstup')} x = {inp.label}</p>
            <div className="table-wrap" tabIndex={0}>
              <table className="t" style={{ maxWidth: 720 }}>
                <thead><tr><th>i</th><th className="num">x_i ({T('activation', 'aktivácia')})</th><th className="num">W[{cp.row}, i] ({T('weight', 'váha')})</th><th className="num">{T('product', 'súčin')}</th></tr></thead>
                <tbody>
                  {inp.x.map((x, i) => (
                    <tr key={i}><td>{i}</td><td className="num">{fnum(x)}</td><td className="num">{fnum(row[i])}</td><td className="num">{fnum(products[i])}</td></tr>
                  ))}
                  <tr><td colSpan={3}><strong>{T('sum = pre-activation', 'súčet = pred-aktivácia')} y<sub>{cp.row}</sub></strong></td><td className="num"><strong>{fnum(sum)}</strong></td></tr>
                </tbody>
              </table>
            </div>
            {gated && !isDown && (
              <div className="callout" style={{ marginTop: 10 }}>
                <div>{T('gate pre-activation', 'pred-aktivácia hradla')} a = {fnum(gated.gPre)} → SiLU(a) = a·σ(a) = {fnum(gated.gPre)} × {fnum(sigmoid(gated.gPre))} = <strong>{fnum(gated.g)}</strong></div>
                <div>{T('up value', 'hodnota up')} u = {fnum(gated.u)}</div>
                <div>{T('gated channel', 'hradlovaný kanál')} z = g × u = <strong>{fnum(gated.z)}</strong></div>
                <VectorView label={T(`column ${ch} of W_down (outgoing weights)`, `stĺpec ${ch} matice W_down (výstupné váhy)`)} values={gated.downCol} kind="par" />
                <VectorView label={T(`contribution of this channel to every output feature: z × W_down[:, ${ch}]`, `príspevok tohto kanála ku každej výstupnej vlastnosti: z × W_down[:, ${ch}]`)} values={gated.downCol.map((v) => v * gated!.z)} />
              </div>
            )}
          </>
        ) : (
          <p className="muted">{T('This tensor is applied elementwise or by lookup, so there is no weighted sum; see Level 0 or the module view.', 'Tento tenzor sa používa po prvkoch alebo vyhľadaním, preto tu nie je vážený súčet; pozri úroveň 0 alebo pohľad na modul.')}</p>
        )}
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={() => go({ lv: 0, c: s.c ?? 0 })}>{T(`Open weight W[${r}, ${s.c ?? 0}] (Level 0)`, `Otvoriť váhu W[${r}, ${s.c ?? 0}] (úroveň 0)`)}</button>
          <button className="ghost" onClick={() => go({ lv: 2 })}>{T('Back to the matrix (Level 2)', 'Späť na maticu (úroveň 2)')}</button>
        </div>
      </div>
    </div>
  );
}

/** LEVEL 0 — one stored scalar: value, sign, precision, coordinate, checkpoint location and provenance. */
export function ScalarView({ p, f, s, go }: DeepProps) {
  const toyApi = useToy();
  const { lang, num: N, fnum, shape } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const flabel = lang === 'sk' ? f.labelSk : f.label;
  const name = tensorName(f, s.b ?? 0, s.e ?? 0);
  const isVec = f.shape.length === 1;
  const r = s.r ?? 0, c = isVec ? null : s.c ?? 0;
  const real = useRealValues(p, name);
  const offsets = useOffsets(p.status === 'official' ? p.id : null);
  const hit = p.status === 'official' ? sampleFor(real.list?.filter((x) => x.modelKey === p.id), name, r, c ?? undefined) : null;
  const realVal = hit?.value ?? null;
  const cp = toyCounterpart(p, f.key, s.b ?? 0, s.e ?? 0, r, c);
  const tt = useMemo(() => (cp ? toyTrace(cp.toy, toyApi.state.seed, toyApi.state.tokens, { variant: toyApi.state.variant, weights: toyApi.weights }) : null), [cp?.toy, toyApi.state.seed, toyApi.state.tokens, toyApi.state.variant, toyApi.weights]); // eslint-disable-line react-hooks/exhaustive-deps
  const toyVal = cp && tt ? toyValue(tt.w, cp) : null;
  const [draft, setDraft] = useState<number | null>(null);
  const fo = offsets.status === 'ok' ? fileOffset(offsets.data, name, r, c) : null;
  const shown = p.status === 'toy' ? toyVal : realVal;
  const bits = shown !== null ? toBf16Bits(shown) : null;
  const parts = bits !== null ? bf16Parts(bits) : null;
  const preview = useMemo(() => {
    if (!cp || !tt || draft === null || !Number.isFinite(draft)) return null;
    const w2 = withEditedWeight(tt.w, cp.name, cp.row, cp.col, draft);
    const a = tt.trace[tt.trace.length - 1].logits;
    const b = forwardFull(cp.toy, w2, toyApi.state.tokens);
    const bl = b[b.length - 1].logits;
    return { before: a, after: bl, diff: maxAbsDiff(a, bl) };
  }, [cp, tt, draft, toyApi.state.tokens]);
  const canCommit = cp && toyApi.state.variant === cp.toy.id;
  return (
    <div className="page wide" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>{T('One stored number:', 'Jedno uložené číslo:')} {flabel} [{r}{c === null ? '' : `, ${c}`}]</h2>
      <div className="grid2">
        <div className="card">
          {lang === 'sk' ? <p><strong>Parameter</strong> je jedno naučené číslo. Toto sa nachádza na súradnici <code>[{r}{c === null ? '' : `, ${c}`}]</code> tenzora <code>{name}</code> {isVec ? '' : `(riadok ${r} = výstupná súradnica, stĺpec ${c} = vstupná vlastnosť)`}.</p> : <p>A <strong>parameter</strong> is one learned scalar. This one sits at coordinate <code>[{r}{c === null ? '' : `, ${c}`}]</code> of <code>{name}</code> {isVec ? '' : `(row ${r} = output coordinate, column ${c} = input feature)`}.</p>}
          {p.status === 'official' ? (
            real.samples.status === 'loading' ? (
              <p className="muted">{T('Checking the shipped real samples for this coordinate…', 'Kontrolujem dodané skutočné vzorky pre túto súradnicu…')}</p>
            ) : realVal !== null ? (
              <>
                <div className="row"><Evidence kind="sample" /><span className="big-num">{lang === 'sk' ? String(realVal).replace('.', ',') : realVal}</span></div>
                <p className="muted" style={{ fontSize: '0.88rem' }}>{T(`Actual value read from ${hit!.sample.shard} of ${hit!.sample.model} @ ${hit!.sample.revision.slice(0, 12)} (raw-bytes SHA-256 ${hit!.sample.rawSha256.slice(0, 16)}…), widened exactly from BF16. ${hit!.sample.note}`, `Skutočná hodnota prečítaná zo súboru ${hit!.sample.shard} modelu ${hit!.sample.model} @ ${hit!.sample.revision.slice(0, 12)} (SHA-256 surových bajtov ${hit!.sample.rawSha256.slice(0, 16)}…), presne rozšírená z BF16. Malý výrez uložených váh. Nie je to celý model, aktivácia ani záznam behu.`)}</p>
              </>
            ) : (
              <div className="callout warn">
                <div className="row"><Evidence kind="unknown" /><strong>{T('Actual value not loaded.', 'Skutočná hodnota nie je načítaná.')}</strong></div>
                <p style={{ marginBottom: 0 }}>{T('Only small real tiles are shipped (e.g. rows 0–15 × cols 0–15 of selected tensors), to avoid downloading multi-gigabyte checkpoints. The structure, location and dtype below are verified; the number itself is not shown rather than invented.', 'Dodávajú sa len malé skutočné výrezy (napr. riadky 0–15 × stĺpce 0–15 vybraných tenzorov), aby nebolo treba sťahovať checkpointy s veľkosťou mnohých gigabajtov. Štruktúra, umiestnenie a typ čísla nižšie sú overené; samotné číslo sa radšej nezobrazí, než aby sa vymyslelo.')} {real.samples.status === 'error' && `(${T('Sample file failed to load', 'Súbor so vzorkami sa nepodarilo načítať')}: ${real.samples.error})`}</p>
                <div className="row" style={{ marginTop: 6 }}>
                  {(real.list ?? []).filter((x) => x.modelKey === p.id).map((x) => {
                    const loc = locateTensor(p, x.tensor);
                    return loc ? (
                      <a key={x.id} className="btn small" href={explorerHref({ m: p.id, lv: 0, t: loc.key, b: loc.b, e: loc.e, r: 0, c: x.logicalShape.length > 1 ? 0 : undefined })}>{T('Real sample', 'Skutočná vzorka')}: {x.tensor.replace('model.', '')}</a>
                    ) : null;
                  })}
                </div>
              </div>
            )
          ) : (
            <><div className="row"><Evidence kind="simulation" /><span className="big-num">{toyVal === null ? '' : fnum(toyVal, 6)}</span></div><p className="faint">{T('Toy simulator value (seeded synthetic weight, float64).', 'Hodnota hračkového simulátora (syntetická váha z osiva, float64).')}</p></>
          )}
          {parts && (
            <div style={{ marginTop: 10 }}>
              <strong>{T('BF16 representation', 'Zápis v BF16')}</strong> {p.status === 'official' && realVal !== null ? <Evidence kind="sample" title="Exact bits of the stored value" /> : <Evidence kind="derived" title="What this toy float64 value would round to in BF16" />}
              <div className="bits" aria-label={`bits ${bits!.toString(2).padStart(16, '0')}`} style={{ margin: '6px 0' }}>
                {bits!.toString(2).padStart(16, '0').split('').map((b, i) => <span key={i} className={i === 0 ? 's' : i < 9 ? 'e' : 'f'}>{b}</span>)}
              </div>
              <div className="faint" style={{ fontSize: '0.84rem' }}>
                {T('sign', 'znamienko')} {parts.sign} ({parts.sign ? T('negative', 'záporné') : T('positive', 'kladné')}) · {T('exponent bits', 'bity exponentu')} {parts.exponentBits} {parts.unbiasedExponent !== null && `(2^${parts.unbiasedExponent})`} · {T('7 fraction bits', '7 bitov zlomku')} {parts.fractionBits} · {T('class', 'trieda')} {parts.kind} · {T('value', 'hodnota')} {parts.value}
                {p.status === 'toy' && shown !== null && !isExactBf16(shown) && T(` (the toy keeps full float64 precision; BF16 would round it to ${parts.value})`, ` (hračka uchováva plnú presnosť float64; BF16 by ju zaokrúhlilo na ${parts.value})`)}
              </div>
              <p className="faint" style={{ fontSize: '0.8rem' }}>{T("BF16 keeps float32's 8-bit exponent (same range) with only 7 fraction bits (about 2–3 significant decimal digits). Arithmetic usually accumulates in float32. This bit picture describes the number format; it is not a dump of file bytes (files store it little-endian).", 'BF16 zachováva 8-bitový exponent formátu float32 (rovnaký rozsah) a má len 7 bitov zlomku (približne 2–3 platné desatinné číslice). Výpočty sa zvyčajne sčítavajú vo float32. Tento obrázok bitov opisuje formát čísla; nie je to výpis bajtov zo súboru (súbory ho ukladajú v poradí little-endian).')}</p>
            </div>
          )}
        </div>
        <div className="card">
          <h3>{T('Where it lives', 'Kde je uložené')}</h3>
          <dl className="kv">
            <dt>{T('Model', 'Model')}</dt><dd>{p.repo ? `${p.repo} @ ${p.revision!.slice(0, 12)}` : p.name}</dd>
            <dt>{T('Tensor', 'Tenzor')}</dt><dd className="mono">{name}</dd>
            <dt>{T('Coordinate', 'Súradnica')}</dt><dd className="mono">[{r}{c === null ? '' : `, ${c}`}] {T('of', 'z')} {shape(f.shape)}</dd>
            <dt>{T('Flat index', 'Plochý index')}</dt><dd className="mono">{N(isVec ? r : r * f.shape[1] + (c ?? 0))} ({T('row-major', 'po riadkoch')})</dd>
            {p.status === 'official' && <><dt>{T('File location', 'Umiestnenie v súbore')}</dt><dd className="mono">{fo ? T(`${fo.shard}, byte ${N(fo.byte)} (2 bytes, little-endian)`, `${fo.shard}, bajt ${N(fo.byte)} (2 bajty, little-endian)`) : offsets.status === 'loading' ? T('loading byte map…', 'načítavam mapu bajtov…') : offsets.status === 'error' ? `${T('byte map unavailable', 'mapa bajtov nie je dostupná')}: ${offsets.error}` : '—'}</dd></>}
            <dt>{T('Runtime', 'Počas behu')}</dt><dd>{T('After loading, the tensor becomes an array in RAM/VRAM. Its memory address is chosen by the allocator at run time and is unrelated to the file offset or the coordinate.', 'Po načítaní sa tenzor stane poľom v RAM/VRAM. Jeho adresu v pamäti zvolí alokátor počas behu a nesúvisí s pozíciou v súbore ani so súradnicou.')}</dd>
          </dl>
          {p.status === 'official' && <p className="faint" style={{ fontSize: '0.8rem' }}>{T('File offset = 8 (length prefix) + header length + tensor data offset + flat index × 2, from the pinned safetensors header.', 'Pozícia v súbore = 8 (predpona dĺžky) + dĺžka hlavičky + posun dát tenzora + plochý index × 2, podľa hlavičky safetensors v pevnej verzii.')} <Evidence kind="derived" /></p>}
          <h3 style={{ marginTop: 14 }}>{T('Change one weight', 'Zmeniť jednu váhu')} {cp?.mapped ? T('(toy analogue)', '(hračkový analóg)') : ''}</h3>
          {cp && toyVal !== null ? (
            <>
              <p className="faint" style={{ fontSize: '0.85rem' }}>{T('Edits apply to the', 'Úpravy sa týkajú tenzora')} <strong>{cp.toy.name}</strong> <code>{cp.name}</code>[{cp.row}{cp.col === null ? '' : `, ${cp.col}`}] ({T('current value', 'aktuálna hodnota')} {fnum(toyVal, 5)}), {T('never to a real checkpoint.', 'nikdy skutočného checkpointu.')}</p>
              <div className="row">
                <NumberField label={T('new value', 'nová hodnota')} value={draft ?? toyVal} onChange={setDraft} step={0.05} width={120} />
                <button onClick={() => setDraft(toyVal + 0.5)}>+0.5</button>
                <button onClick={() => setDraft(null)} className="ghost">{T('clear', 'zrušiť')}</button>
              </div>
              {preview && (
                <div className="banner-edit" style={{ marginTop: 8 }}>
                  {T('Preview (not yet applied): last-position logits change by up to', 'Náhľad (zatiaľ nepoužité): logity poslednej pozície sa zmenia najviac o')} <strong>{fnum(preview.diff, 4)}</strong>.
                  <VectorView label={T('logits before', 'logity pred')} values={preview.before} />
                  <VectorView label={T('logits after', 'logity po')} values={preview.after} />
                  {canCommit ? (
                    <button className="small" style={{ marginTop: 6 }} onClick={() => { toyApi.dispatch({ type: 'edit', name: cp.name, row: cp.row, col: cp.col, value: draft! }); setDraft(null); }}>{T('Apply to the shared toy simulator (marks it “edited”, invalidates its cache)', 'Použiť v spoločnom hračkovom simulátore (označí ho ako „upravený“, zneplatní jeho cache)')}</button>
                  ) : (
                    <button className="small" style={{ marginTop: 6 }} onClick={() => toyApi.dispatch({ type: 'variant', v: cp.toy.id as 'toy-dense' | 'toy-moe' })}>{T(`Switch shared simulator to ${cp.toy.name} first`, `Najprv prepnúť spoločný simulátor na ${cp.toy.name}`)}</button>
                  )}
                </div>
              )}
              {toyApi.edited && <div className="banner-edit" style={{ marginTop: 8 }}>{T('The shared toy simulator is in an', 'Spoločný hračkový simulátor je v')} <strong>{T('edited state', 'upravenom stave')}</strong> ({toyApi.state.edits.length}). <button className="small" onClick={() => toyApi.dispatch({ type: 'resetWeights' })}>{T('Reset to seeded fixture', 'Obnoviť pôvodné váhy z osiva')}</button></div>}
            </>
          ) : <p className="muted">{T('No toy analogue for this tensor.', 'Pre tento tenzor neexistuje hračkový analóg.')}</p>}
          <div className="row" style={{ marginTop: 10 }}>
            {!isVec && <button className="ghost" onClick={() => go({ lv: 1 })}>{T('Up: the unit using this weight (Level 1)', 'Hore: jednotka, ktorá túto váhu používa (úroveň 1)')}</button>}
            <button className="ghost" onClick={() => go({ lv: 2 })}>{T('Up: the whole matrix (Level 2)', 'Hore: celá matica (úroveň 2)')}</button>
            <ExportButton filename={`${p.id}-scalar.json`} data={() => ({ model: p.repo ?? p.name, revision: p.revision ?? null, tensor: name, coordinate: c === null ? [r] : [r, c], dtype: p.status === 'official' ? p.dtype : 'float64 (toy)', value: shown, evidence: p.status === 'toy' ? 'illustrative-simulation' : realVal !== null ? 'actual-checkpoint-sample' : 'unknown (value not loaded)', fileLocation: fo, sampleSha256: hit?.sample.rawSha256 ?? null })} />
          </div>
        </div>
      </div>
    </div>
  );
}

/** LEVEL 6 — execution & learning for the selected model. */
export function ExecutionView({ p }: { p: CoreProfile }) {
  const a = accounting(p);
  const { lang, num: N } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const kv = kvBytesPerToken(p);
  const steps = [
    [T('Messages → chat template', 'Správy → šablóna rozhovoru'), T('Role markers and (optionally) an empty <think></think> block are inserted as text.', 'Označenia rolí a (voliteľne) prázdny blok <think></think> sa vložia ako text.'), '#/pipeline'],
    [T('Tokenizer → token ids', 'Tokenizér → ID tokenov'), T(`Byte-level BPE; ${p.tokenizerVocab ? N(p.tokenizerVocab) : p.vocabRows} ids defined.`, `Byte-level BPE; definovaných ${p.tokenizerVocab ? N(p.tokenizerVocab) : p.vocabRows} ID.`), '#/pipeline'],
    [T('Embedding row lookup', 'Vyhľadanie riadku embeddingu'), T(`${N(p.d)} numbers per token copied from the table.`, `${N(p.d)} čísel na token sa skopíruje z tabuľky.`), explorerHref({ m: p.id, lv: 3, mod: 'embedding' })],
    ['Prefill', T(`All prompt positions go through block 0, then block 1, … block ${p.layers - 1}. Positions run in parallel inside a block; blocks run in sequence. Each block stores K and V for every position.`, `Všetky pozície zadania prejdú blokom 0, potom blokom 1, … blokom ${p.layers - 1}. V rámci bloku bežia pozície súčasne; bloky idú za sebou. Každý blok si uloží K a V pre každú pozíciu.`), '#/labs/cache'],
    ['Decode', T(`One new token at a time passes through all ${p.layers} blocks, attends to cached K/V, and appends its own K/V (${fmtBytes(kv)} per token at BF16 for this model, K/V only).`, `Po jednom novom tokene prechádza všetkými ${p.layers} blokmi, číta uložené K/V a pripojí vlastné K/V (${fmtBytes(kv)} na token v BF16 pre tento model, len K/V).`), '#/labs/cache'],
    [T('Final norm → logits', 'Záverečná norma → logity'), T(`${N(p.vocabRows)} logits from the untied head.`, `${N(p.vocabRows)} logitov z neviazanej hlavy.`), explorerHref({ m: p.id, lv: 3, mod: 'lm_head' })],
    [T('Sampling → next token', 'Vzorkovanie → ďalší token'), T('Temperature / top-k / top-p / draw; the token is appended and the loop repeats.', 'Teplota / top-k / top-p / výber; token sa pripojí a slučka sa opakuje.'), '#/labs/sampling'],
    [T('Training path (separate)', 'Trénovanie (samostatne)'), T('Targets = inputs shifted by one; cross-entropy loss; gradients by the chain rule; optimizer updates weights. Inference never changes weights.', 'Ciele = vstupy posunuté o jeden; chyba krížovej entropie; gradienty podľa reťazového pravidla; optimalizátor upraví váhy. Inferencia váhy nikdy nemení.'), '#/labs/training'],
    [T('Memory & compute', 'Pamäť a výpočty'), T(`${fmtBytes(a.total * 2)} of BF16 weights before any cache or workspace.`, `${fmtBytes(a.total * 2)} váh v BF16 ešte pred akoukoľvek cache či pracovným priestorom.`), '#/labs/memory'],
  ];
  return (
    <div className="page" style={{ paddingTop: 12 }}>
      <h2 style={{ marginTop: 0 }}>{T('Execution and learning', 'Beh a učenie')} · {p.name}</h2>
      <ol style={{ paddingLeft: 20 }}>
        {steps.map(([t, d, h]) => (
          <li key={t} style={{ marginBottom: 10 }}><a href={h}><strong>{t}</strong></a> — <span className="muted">{d}</span></li>
        ))}
      </ol>
      <div className="callout">{T(`Each step links to a working laboratory driven by the toy simulator, or to real tokenizer data. A real ${p.name} is not running in your browser.`, `Každý krok vedie do funkčného laboratória poháňaného hračkovým simulátorom alebo k skutočným údajom tokenizéra. Skutočný ${p.name} vo vašom prehliadači nebeží.`)}</div>
    </div>
  );
}
