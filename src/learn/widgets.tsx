import { useMemo, useState, type ReactElement } from 'react';
import { useLang } from '../i18n/i18n';
import { useToy } from '../state/toyStore';
import { TOY_VOCAB } from '../math/toy';
import { sigmoid, silu, stableSoftmax, rmsnorm, routeFromLogits } from '../math/ops';
import { samplingStages, drawToken } from '../math/sampling';
import { makeRng } from '../math/rng';
import { bf16Parts, toBf16Bits } from '../math/bf16';
import { fileOffset, sampleFor, useOffsets, useSamples, useTokenizerExamples } from '../data/data';
import { QWEN3_30B_A3B, QWEN3_8B } from '../models/profiles';
import { accounting } from '../models/shapes';
import { GB, kvCacheBytes, weightBytes } from '../math/memory';
import type { WidgetId } from '../content/chapters';
import { Frame, HBars, Range, pick } from './kit';
import { REGISTRY2 } from './widgets2';

// Small, self-contained experiments for the Learn chapters. All labels exist in English and Slovak.

// 1 ---------------------------------------------------------------------------
function NextToken() {
  const { lang, num } = useLang();
  const { state, trace, dispatch } = useToy();
  const probs = stableSoftmax(trace[trace.length - 1].logits).probs;
  const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 6);
  return (
    <Frame title={pick(lang, 'probabilities of the next word', 'pravdepodobnosti ďalšieho slova')} ev={['simulation']} note={pick(lang, 'A tiny toy model with 12 English toy words and random (untrained) weights, so its guesses are not meaningful. The point is the mechanism: text in, one probability per possible next word out.', 'Malý hračkový model s 12 anglickými hračkovými slovami a náhodnými (netrénovanými) váhami, preto jeho odhady nemajú zmysel. Ide o mechanizmus: dnu vstúpi text, von vyjde jedna pravdepodobnosť pre každé možné ďalšie slovo.')}>
      <div className="row" style={{ gap: 6 }}>
        {state.tokens.map((t, i) => (
          <select key={i} aria-label={pick(lang, `word ${i + 1}`, `slovo ${i + 1}`)} value={t} onChange={(e) => dispatch({ type: 'tokens', tokens: state.tokens.map((o, k) => (k === i ? Number(e.target.value) : o)) })}>
            {TOY_VOCAB.map((w, id) => <option key={id} value={id}>{w}</option>)}
          </select>
        ))}
        <span className="faint">→ ?</span>
      </div>
      <div style={{ marginTop: 8 }}>
        <HBars items={order.map(([p, i]) => ({ label: TOY_VOCAB[i], v: p, extra: `${num(p * 100, 1)} %` }))} max={1} />
      </div>
    </Frame>
  );
}

// 2 ---------------------------------------------------------------------------
function RealWeight() {
  const { lang, num } = useLang();
  const s = useSamples();
  const off = useOffsets('qwen3-8b');
  const name = 'model.layers.0.self_attn.q_proj.weight';
  const hit = s.status === 'ok' ? sampleFor(s.data.samples.filter((x) => x.modelKey === 'qwen3-8b'), name, 0, 0) : null;
  const v = hit?.value ?? null;
  const fo = off.status === 'ok' ? fileOffset(off.data, name, 0, 0) : null;
  const parts = v !== null ? bf16Parts(toBf16Bits(v)) : null;
  return (
    <Frame title={pick(lang, 'a real weight from Qwen3-8B', 'skutočná váha z Qwen3-8B')} ev={['sample', 'verified']}>
      {s.status === 'loading' && <p className="muted">{pick(lang, 'Loading the real sample…', 'Načítavam skutočnú vzorku…')}</p>}
      {s.status === 'error' && <p className="muted">{pick(lang, 'The real sample could not be loaded.', 'Skutočnú vzorku sa nepodarilo načítať.')}</p>}
      {v !== null && parts && (
        <>
          <div className="big-num" style={{ fontSize: '1.6rem' }}>{v.toString().replace('.', lang === 'sk' ? ',' : '.')}</div>
          <div className="bits" style={{ margin: '8px 0' }} aria-label="BF16 bits">
            {parts.bits.toString(2).padStart(16, '0').split('').map((bit, i) => <span key={i} className={i === 0 ? 's' : i < 9 ? 'e' : 'f'} title={i === 0 ? pick(lang, 'sign', 'znamienko') : i < 9 ? pick(lang, 'exponent', 'exponent') : pick(lang, 'fraction', 'zlomok')}>{bit}</span>)}
          </div>
          <div className="faint" style={{ fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--danger)' }}>■</span> {pick(lang, 'sign', 'znamienko')} {parts.sign ? '−' : '+'} · <span style={{ color: 'var(--norm)' }}>■</span> {pick(lang, 'exponent', 'exponent')} 2^{parts.unbiasedExponent} · <span style={{ color: 'var(--accent)' }}>■</span> {pick(lang, '7 fraction bits', '7 bitov zlomku')}
          </div>
          <dl className="kv" style={{ marginTop: 8, fontSize: '0.88rem' }}>
            <dt>{pick(lang, 'tensor', 'tenzor')}</dt><dd className="mono">{name} [0, 0]</dd>
            <dt>{pick(lang, 'file', 'súbor')}</dt><dd className="mono">{fo ? `${fo.shard}, ${pick(lang, 'byte', 'bajt')} ${num(fo.byte)}` : '…'}</dd>
            <dt>{pick(lang, 'meaning', 'význam')}</dt><dd>{pick(lang, 'row 0 = query output 0 of block 0; column 0 = input feature 0', 'riadok 0 = výstup dopytu 0 v bloku 0; stĺpec 0 = vstupná vlastnosť 0')}</dd>
          </dl>
        </>
      )}
    </Frame>
  );
}

// 3 ---------------------------------------------------------------------------
function Neuron() {
  const { lang, num } = useLang();
  const [x, setX] = useState([1, 0.5, -1]);
  const [w, setW] = useState([0.8, -1.2, 0.4]);
  const [act, setAct] = useState<'silu' | 'none'>('silu');
  const prods = x.map((xi, i) => xi * w[i]);
  const sum = prods.reduce((a, b2) => a + b2, 0);
  const y = act === 'silu' ? silu(sum) : sum;
  const W = 420, H = 190;
  return (
    <Frame title={pick(lang, 'one neuron with three inputs', 'jeden neurón s tromi vstupmi')} ev={['simulation']} note={pick(lang, 'Line thickness shows the size of a weight; solid lines are positive, dashed lines negative.', 'Hrúbka čiary ukazuje veľkosť váhy; plné čiary sú kladné, prerušované záporné.')}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 520 }} role="img" aria-label={pick(lang, `Neuron: output ${y.toFixed(2)}`, `Neurón: výstup ${y.toFixed(2)}`)}>
        {x.map((xi, i) => {
          const cy = 35 + i * 60;
          return (
            <g key={i}>
              <line x1={70} y1={cy} x2={260} y2={95} stroke={w[i] >= 0 ? 'var(--accent)' : 'var(--danger)'} strokeWidth={1 + Math.abs(w[i]) * 3} strokeDasharray={w[i] < 0 ? '6 4' : undefined} />
              <circle cx={48} cy={cy} r={22} fill="var(--bg)" stroke="var(--act)" strokeWidth={2} />
              <text x={48} y={cy + 4} textAnchor="middle" fontSize={12} fill="var(--text)">{num(xi, 1)}</text>
              <text x={150} y={cy + (95 - cy) * 0.45 - 6} fontSize={11} fill="var(--weight)">w{i + 1}={num(w[i], 1)}</text>
            </g>
          );
        })}
        <circle cx={290} cy={95} r={34} fill="color-mix(in srgb, var(--weight) 20%, var(--panel))" stroke="var(--weight)" strokeWidth={2} />
        <text x={290} y={91} textAnchor="middle" fontSize={11} fill="var(--text)">Σ = {num(sum, 2)}</text>
        <text x={290} y={106} textAnchor="middle" fontSize={10} fill="var(--text-2)">{act === 'silu' ? 'SiLU' : pick(lang, 'no f', 'bez f')}</text>
        <line x1={324} y1={95} x2={370} y2={95} stroke="var(--act)" strokeWidth={2} />
        <circle cx={392} cy={95} r={22} fill="var(--bg)" stroke="var(--act)" strokeWidth={2} />
        <text x={392} y={99} textAnchor="middle" fontSize={12} fill="var(--text)">{num(y, 2)}</text>
      </svg>
      <div className="grid2">
        <div>{x.map((v, i) => <Range key={i} label={`${pick(lang, 'input', 'vstup')} x${i + 1}`} value={v} min={-2} max={2} onChange={(nv) => setX(x.map((o, k) => (k === i ? nv : o)))} />)}</div>
        <div>{w.map((v, i) => <Range key={i} label={`${pick(lang, 'weight', 'váha')} w${i + 1}`} value={v} min={-2} max={2} onChange={(nv) => setW(w.map((o, k) => (k === i ? nv : o)))} />)}</div>
      </div>
      <div className="row" role="group" aria-label={pick(lang, 'activation function', 'aktivačná funkcia')}>
        <button className="small" aria-pressed={act === 'silu'} onClick={() => setAct('silu')}>SiLU</button>
        <button className="small" aria-pressed={act === 'none'} onClick={() => setAct('none')}>{pick(lang, 'no activation', 'bez aktivácie')}</button>
      </div>
      <p className="mono" style={{ fontSize: '0.88rem' }}>
        {prods.map((_, i) => `${num(x[i], 1)}×${num(w[i], 1)}`).join(' + ')} = {num(sum, 3)} → {act === 'silu' ? `SiLU(${num(sum, 3)})` : ''} = <strong>{num(y, 3)}</strong>
      </p>
    </Frame>
  );
}

// 4 ---------------------------------------------------------------------------
function SiluPlot() {
  const { lang, num } = useLang();
  const [a, setA] = useState(1.5);
  const W = 360, H = 200, sx = (v: number) => 20 + ((v + 6) / 12) * (W - 40), sy = (v: number) => H - 20 - ((v + 1) / 7) * (H - 40);
  const pts = Array.from({ length: 121 }, (_, i) => -6 + i * 0.1).map((v) => `${sx(v)},${sy(silu(v))}`).join(' ');
  return (
    <Frame title={pick(lang, 'the SiLU activation function', 'aktivačná funkcia SiLU')} ev={['verified']} note={pick(lang, 'Qwen3 uses SiLU(a) = a · σ(a), where σ is the logistic “S-curve” between 0 and 1.', 'Qwen3 používa SiLU(a) = a · σ(a), kde σ je logistická „S-krivka“ medzi 0 a 1.')}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 440 }} role="img" aria-label="SiLU">
        <line x1={sx(-6)} y1={sy(0)} x2={sx(6)} y2={sy(0)} stroke="var(--line)" />
        <line x1={sx(0)} y1={sy(-1)} x2={sx(0)} y2={sy(6)} stroke="var(--line)" />
        <line x1={sx(-1)} y1={sy(-1)} x2={sx(6)} y2={sy(6)} stroke="var(--text-3)" strokeDasharray="4 4" />
        <polyline points={pts} fill="none" stroke="var(--ffn)" strokeWidth={3} />
        <circle cx={sx(a)} cy={sy(silu(a))} r={6} fill="var(--focus)" />
        <text x={sx(-5.8)} y={sy(5.6)} fontSize={10} fill="var(--text-3)">{pick(lang, 'dashed: output = input', 'prerušovaná: výstup = vstup')}</text>
      </svg>
      <Range label="a" value={a} min={-6} max={6} onChange={setA} />
      <p className="mono">σ({num(a, 2)}) = {num(sigmoid(a), 3)} → SiLU = {num(a, 2)} × {num(sigmoid(a), 3)} = <strong>{num(silu(a), 3)}</strong></p>
    </Frame>
  );
}

// 5 ---------------------------------------------------------------------------
const MV = [[0.5, -0.2, 0.1, 0.9], [-0.7, 0.4, 0.3, 0.0], [0.2, 0.2, -0.6, 0.5]];
function MatVec() {
  const { lang, num } = useLang();
  const [x, setX] = useState([1, 2, -1, 0.5]);
  const [row, setRow] = useState(0);
  const y = MV.map((r) => r.reduce((s, w, i) => s + w * x[i], 0));
  return (
    <Frame title={pick(lang, 'three neurons as one matrix', 'tri neuróny ako jedna matica')} ev={['simulation']} note={pick(lang, 'Teaching values chosen for the example. Click a row to see its weighted sum.', 'Hodnoty zvolené pre príklad. Kliknite na riadok a uvidíte jeho vážený súčet.')}>
      <div className="row top" style={{ gap: 18 }}>
        <div>
          <div className="faint" style={{ fontSize: '0.8rem' }}>W [3 × 4] ({pick(lang, 'rows = neurons', 'riadky = neuróny')})</div>
          <div className="matrix-grid" style={{ gridTemplateColumns: 'repeat(4, 52px)' }}>
            {MV.map((r, i) => r.map((v, j) => <button key={`${i}${j}`} className={`cell${row === i ? ' hl' : ''}`} onClick={() => setRow(i)} aria-label={`W[${i}][${j}] = ${v}`}>{num(v, 1)}</button>))}
          </div>
        </div>
        <div style={{ fontSize: '1.4rem', alignSelf: 'center' }}>×</div>
        <div>
          <div className="faint" style={{ fontSize: '0.8rem' }}>x [4]</div>
          {x.map((v, i) => <div key={i} className="chip" style={{ display: 'flex', marginBottom: 2 }}>{num(v, 1)}</div>)}
        </div>
        <div style={{ fontSize: '1.4rem', alignSelf: 'center' }}>=</div>
        <div>
          <div className="faint" style={{ fontSize: '0.8rem' }}>y [3]</div>
          {y.map((v, i) => <button key={i} className={`chip${row === i ? ' sel' : ''}`} style={{ display: 'flex', marginBottom: 2 }} onClick={() => setRow(i)}>{num(v, 2)}</button>)}
        </div>
      </div>
      <p className="mono" style={{ fontSize: '0.86rem' }}>y[{row}] = {MV[row].map((w, i) => `${num(w, 1)}×${num(x[i], 1)}`).join(' + ')} = <strong>{num(y[row], 2)}</strong></p>
      <div className="row">{x.map((v, i) => <Range key={i} label={`x${i}`} value={v} min={-2} max={2} step={0.5} onChange={(nv) => setX(x.map((o, k) => (k === i ? nv : o)))} />)}</div>
      <p className="faint" style={{ fontSize: '0.85rem' }}>{pick(lang, `This matrix holds 3 × 4 = 12 weights. Qwen3-8B’s W_Q has the same layout with 4,096 × 4,096 = ${(4096 * 4096).toLocaleString('en-US')}.`, `Táto matica má 3 × 4 = 12 váh. W_Q v Qwen3-8B má rovnaké usporiadanie s 4 096 × 4 096 = ${(4096 * 4096).toLocaleString('sk-SK')} váhami.`)}</p>
    </Frame>
  );
}

// 6 ---------------------------------------------------------------------------
function Tokens() {
  const { lang } = useLang();
  const tok = useTokenizerExamples();
  const [id, setId] = useState('raw-slovak-2');
  const [sel, setSel] = useState(0);
  if (tok.status !== 'ok') return <Frame title={pick(lang, 'real Qwen3 tokens', 'skutočné tokeny Qwen3')} ev={['sample']}><p className="muted">{tok.status === 'error' ? pick(lang, 'Could not load the tokenizer examples.', 'Príklady tokenizácie sa nepodarilo načítať.') : '…'}</p></Frame>;
  const raw = tok.data.examples.filter((e) => e.kind === 'raw');
  const e = raw.find((x) => x.id === id) ?? raw[0];
  const piece = (k: number) => (e.offsets ? e.renderedText.slice(e.offsets[k][0], e.offsets[k][1]) : e.decodedPieces[k]);
  const partial = (k: number) => e.decodedPieces[k].includes('�');
  const i = Math.min(sel, e.ids.length - 1);
  return (
    <Frame title={pick(lang, 'real Qwen3 tokens', 'skutočné tokeny Qwen3')} ev={['sample']} note={pick(lang, 'Actual output of the pinned Qwen3 tokenizer (computed offline). No model was run.', 'Skutočný výstup tokenizéra Qwen3 v pevnej verzii (vypočítaný vopred). Žiadny model sa nespustil.')}>
      <div className="row" role="group" aria-label={pick(lang, 'sentence', 'veta')}>
        {raw.map((x) => <button key={x.id} className="small" aria-pressed={x.id === e.id} onClick={() => { setId(x.id); setSel(0); }}>{x.text}</button>)}
      </div>
      <div className="row" style={{ gap: 4, marginTop: 8 }}>
        {e.ids.map((tid, k) => (
          <button key={k} className={`chip${k === i ? ' sel' : ''}${partial(k) ? ' special' : ''}`} onClick={() => setSel(k)} aria-label={`token ${k}: ${JSON.stringify(piece(k))}, id ${tid}`}>
            <span>{piece(k).replace(/ /g, '␣')}</span><span className="faint">{tid}</span>
          </button>
        ))}
      </div>
      <p style={{ fontSize: '0.9rem', marginTop: 8 }}>
        {pick(lang, `${e.ids.length} tokens. Selected token id ${e.ids[i]} covers “${piece(i)}”.`, `${e.ids.length} tokenov. Vybraný token s ID ${e.ids[i]} pokrýva „${piece(i)}“.`)}{' '}
        {partial(i) && pick(lang, 'It is only one byte of a two-byte character: together with its neighbour it forms the letter.', 'Je to len jeden bajt dvojbajtového znaku: písmeno vytvorí až spolu so susedným tokenom.')}
      </p>
      <p className="faint" style={{ fontSize: '0.85rem' }}>{pick(lang, `Next, row ${e.ids[i]} of the embedding table (4,096 numbers) is copied as this token’s starting vector.`, `Potom sa riadok ${e.ids[i]} tabuľky embeddingov (4 096 čísel) skopíruje ako počiatočný vektor tohto tokenu.`)}</p>
    </Frame>
  );
}

// 7 ---------------------------------------------------------------------------
function Rms() {
  const { lang, num } = useLang();
  const [x, setX] = useState([3, -1, 0.5, 2, -2.5, 1]);
  const [add, setAdd] = useState(true);
  const corr = [0.2, -0.4, 0.1, 0, 0.3, -0.2];
  const ms = x.reduce((s, v) => s + v * v, 0) / x.length;
  const n = rmsnorm(x, x.map(() => 1), 1e-6);
  const after = x.map((v, i) => v + (add ? corr[i] : 0));
  return (
    <Frame title={pick(lang, 'RMSNorm and the residual addition', 'RMSNorm a reziduálne sčítanie')} ev={['simulation']} note={pick(lang, 'γ = 1 here; in Qwen every position has its own learned γ.', 'Tu je γ = 1; v Qwen má každá pozícia vlastné naučené γ.')}>
      <div className="row">{x.map((v, i) => <Range key={i} label={`x${i}`} value={v} min={-4} max={4} step={0.5} onChange={(nv) => setX(x.map((o, k) => (k === i ? nv : o)))} />)}</div>
      <p className="mono" style={{ fontSize: '0.88rem' }}>RMS = √({num(ms, 3)} + ε) = {num(Math.sqrt(ms), 3)}</p>
      <div className="grid2">
        <div>
          <div className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, 'before', 'pred')}</div>
          <HBars items={x.map((v, i) => ({ label: `x${i}`, v }))} max={4} />
        </div>
        <div>
          <div className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, 'after RMSNorm (typical size ≈ 1)', 'po RMSNorm (typická veľkosť ≈ 1)')}</div>
          <HBars items={n.map((v, i) => ({ label: `x${i}/RMS`, v }))} max={4} />
        </div>
      </div>
      <label className="field inline" style={{ marginTop: 8 }}><input type="checkbox" checked={add} onChange={(e) => setAdd(e.target.checked)} /> {pick(lang, 'add a block’s correction to the stream (x + correction)', 'pripočítať opravu bloku k prúdu (x + oprava)')}</label>
      <p className="mono" style={{ fontSize: '0.85rem' }}>{after.map((v) => num(v, 1)).join('  ')}</p>
    </Frame>
  );
}

// 8 ---------------------------------------------------------------------------
function AttentionMini() {
  const { lang, num } = useLang();
  const { state, trace } = useToy();
  const T = state.tokens.length;
  const [q, setQ] = useState(T - 1);
  const t = Math.min(q, T - 1);
  const pr = trace[t].layers[0].attn.probs[0];
  return (
    <Frame title={pick(lang, 'which earlier words a word looks at', 'na ktoré predchádzajúce slová sa slovo pozerá')} ev={['simulation']} note={pick(lang, 'Computed by the toy model (block 0, head 0) with random weights; the pattern itself is meaningless, the mechanism is real. Future words always get exactly 0 %.', 'Vypočítané hračkovým modelom (blok 0, hlava 0) s náhodnými váhami; samotný vzor nemá význam, mechanizmus je skutočný. Budúce slová dostanú vždy presne 0 %.')}>
      <div className="row" style={{ gap: 4 }}>
        {state.tokens.map((tk, i) => <button key={i} className={`chip${i === t ? ' sel' : ''}`} onClick={() => setQ(i)}>{TOY_VOCAB[tk]}</button>)}
      </div>
      <p className="faint" style={{ fontSize: '0.85rem' }}>{pick(lang, `Query word: “${TOY_VOCAB[state.tokens[t]]}” (position ${t}). Click another word.`, `Slovo s dopytom: „${TOY_VOCAB[state.tokens[t]]}“ (pozícia ${t}). Kliknite na iné slovo.`)}</p>
      <HBars items={pr.map((p, i) => ({ label: `${i}: ${TOY_VOCAB[state.tokens[i]]}`, v: p, on: i <= t, extra: i > t ? pick(lang, 'future', 'budúce') : `${num(p * 100, 1)} %` }))} max={1} />
    </Frame>
  );
}

// 9 ---------------------------------------------------------------------------
function Gated() {
  const { lang, num } = useLang();
  const [x, setX] = useState(1);
  const [wg, setWg] = useState(1.5);
  const [wu, setWu] = useState(0.8);
  const a = wg * x, u = wu * x, g = silu(a), z = g * u;
  return (
    <Frame title={pick(lang, 'one SwiGLU gated channel', 'jeden hradlovaný kanál SwiGLU')} ev={['simulation']} note={pick(lang, 'Simplified to a single input number; in Qwen each sum runs over 4,096 inputs (2,048 in Qwen3-30B-A3B).', 'Zjednodušené na jedno vstupné číslo; v Qwen každý súčet prebieha cez 4 096 vstupov (2 048 v Qwen3-30B-A3B).')}>
      <div className="row">
        <Range label={pick(lang, 'input x', 'vstup x')} value={x} min={-2} max={2} onChange={setX} />
        <Range label="w_gate" value={wg} min={-2} max={2} onChange={setWg} />
        <Range label="w_up" value={wu} min={-2} max={2} onChange={setWu} />
      </div>
      <HBars max={4} items={[
        { label: 'gate a', v: a, extra: num(a, 3) },
        { label: 'SiLU(a)', v: g, extra: num(g, 3) },
        { label: 'up u', v: u, extra: num(u, 3) },
        { label: 'z = g·u', v: z, extra: num(z, 3) },
      ]} />
      <p style={{ fontSize: '0.9rem' }}>{a < -2 ? pick(lang, 'The gate is strongly negative, so SiLU is almost 0: the channel is nearly closed whatever “up” says.', 'Hradlo je silne záporné, SiLU je takmer 0: kanál je takmer zatvorený bez ohľadu na hodnotu up.') : pick(lang, 'Move w_gate to the left to close the gate; the output z follows.', 'Posuňte w_gate doľava, aby ste hradlo zatvorili; výstup z sa prispôsobí.')}</p>
    </Frame>
  );
}

// 10 --------------------------------------------------------------------------
function RouterMini() {
  const { lang, num } = useLang();
  const [logits, setLogits] = useState([1.2, -0.3, 0.4, 2.0, -1.0, 0.8, 0.1, -0.6]);
  const [k, setK] = useState(2);
  const r = routeFromLogits(logits, k, true);
  return (
    <Frame title={pick(lang, 'a router choosing experts', 'smerovač vyberajúci expertov')} ev={['simulation']} note={pick(lang, `Toy: 8 experts, top-${k}. Qwen3-30B-A3B scores 128 experts and keeps the top 8 in every block, for every token.`, `Hračka: 8 expertov, top-${k}. Qwen3-30B-A3B hodnotí 128 expertov a v každom bloku ponechá pre každý token 8 najlepších.`)}>
      <div className="row">{logits.map((v, i) => <Range key={i} label={`E${i}`} value={v} min={-3} max={3} step={0.1} onChange={(nv) => setLogits(logits.map((o, j) => (j === i ? nv : o)))} />)}</div>
      <Range label={pick(lang, 'experts kept (k)', 'ponechaní experti (k)')} value={k} min={1} max={4} step={1} onChange={setK} fmt={(v) => String(v)} />
      <div className="grid2">
        <div>
          <div className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, 'probability of each expert', 'pravdepodobnosť každého experta')}</div>
          <HBars max={1} items={r.probs.map((p, i) => ({ label: `E${i}`, v: p, on: r.selected.includes(i), extra: `${num(p * 100, 1)} %` }))} />
        </div>
        <div>
          <div className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, 'selected, renormalized to 100 %', 'vybraní, prepočítaní na 100 %')}</div>
          <HBars max={1} items={r.selected.map((e, i) => ({ label: `E${e}`, v: r.alpha[i], extra: `${num(r.alpha[i] * 100, 1)} %` }))} />
        </div>
      </div>
    </Frame>
  );
}

// 11 --------------------------------------------------------------------------
function Stack() {
  const { lang, num } = useLang();
  const { trace, state, profile } = useToy();
  const P = trace[trace.length - 1];
  const stages = [
    { label: pick(lang, `embedding row for “${TOY_VOCAB[P.token]}”`, `riadok embeddingu pre „${TOY_VOCAB[P.token]}“`), v: P.embed },
    ...P.layers.flatMap((L, l) => [
      { label: pick(lang, `block ${l}: + attention`, `blok ${l}: + pozornosť`), v: L.h },
      { label: pick(lang, `block ${l}: + ${profile.kind === 'dense' ? 'FFN' : 'experts'}`, `blok ${l}: + ${profile.kind === 'dense' ? 'FFN' : 'experti'}`), v: L.out },
    ]),
    { label: pick(lang, 'final RMSNorm', 'záverečná RMSNorm'), v: P.finalNorm },
  ];
  const [s, setS] = useState(0);
  const cur = stages[Math.min(s, stages.length - 1)];
  const probs = stableSoftmax(P.logits).probs;
  const best = probs.indexOf(Math.max(...probs));
  return (
    <Frame title={pick(lang, 'one token’s vector through the whole toy model', 'vektor jedného tokenu celým hračkovým modelom')} ev={['simulation']} note={pick(lang, `The toy model has ${profile.layers} blocks and vectors of 8 numbers; Qwen3-8B has 36 blocks and 4,096 numbers.`, `Hračkový model má ${profile.layers} bloky a vektory s 8 číslami; Qwen3-8B má 36 blokov a 4 096 čísel.`)}>
      <div className="timeline">{stages.map((st, i) => <button key={i} className={`step${i === s ? ' cur' : i < s ? ' done' : ''}`} onClick={() => setS(i)}>{i + 1}. {st.label}</button>)}</div>
      <div className="row" style={{ margin: '8px 0' }}>
        <button className="small" disabled={s === 0} onClick={() => setS(s - 1)}>{pick(lang, '◀ back', '◀ späť')}</button>
        <button className="small" disabled={s >= stages.length - 1} onClick={() => setS(s + 1)}>{pick(lang, 'next ▶', 'ďalej ▶')}</button>
      </div>
      <HBars max={Math.max(...stages.flatMap((x) => x.v.map(Math.abs)))} items={cur.v.map((v, i) => ({ label: `#${i}`, v, extra: num(v, 2) }))} />
      {s === stages.length - 1 && <p>{pick(lang, `The output head turns this vector into scores; the most likely next toy word is “${TOY_VOCAB[best]}” (${num(probs[best] * 100, 1)} %).`, `Výstupná hlava premení tento vektor na skóre; najpravdepodobnejšie ďalšie hračkové slovo je „${TOY_VOCAB[best]}“ (${num(probs[best] * 100, 1)} %).`)}</p>}
      <p className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, `Prompt: ${state.tokens.map((x) => TOY_VOCAB[x]).join(' ')}`, `Zadanie: ${state.tokens.map((x) => TOY_VOCAB[x]).join(' ')}`)}</p>
    </Frame>
  );
}

// 12 --------------------------------------------------------------------------
function SamplingMini() {
  const { lang, num } = useLang();
  const { trace } = useToy();
  const [T, setT] = useState(1);
  const [seed, setSeed] = useState(1);
  const [drawn, setDrawn] = useState<string[]>([]);
  const st = samplingStages(trace[trace.length - 1].logits, { temperature: T, topK: 0, topP: 1 });
  const top = st.order.slice(0, 6);
  return (
    <Frame title={pick(lang, 'temperature and a random draw', 'teplota a náhodný výber')} ev={['simulation']}>
      <Range label={pick(lang, 'temperature', 'teplota')} value={T} min={0} max={2} step={0.05} onChange={setT} fmt={(v) => (v === 0 ? pick(lang, '0 (always the most likely)', '0 (vždy najpravdepodobnejšie)') : num(v, 2))} />
      <HBars max={1} items={top.map((i) => ({ label: TOY_VOCAB[i], v: st.final[i], extra: `${num(st.final[i] * 100, 1)} %` }))} />
      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={() => { const r = drawToken(st.final, makeRng(seed)); setSeed(seed + 1); setDrawn([TOY_VOCAB[r.token], ...drawn].slice(0, 12)); }}>{pick(lang, 'Draw a word', 'Vylosovať slovo')}</button>
        <button className="ghost" onClick={() => setDrawn([])}>{pick(lang, 'clear', 'vymazať')}</button>
      </div>
      <p className="mono" style={{ fontSize: '0.88rem' }}>{drawn.join(', ') || pick(lang, '(no draws yet)', '(zatiaľ nič)')}</p>
      <p className="faint" style={{ fontSize: '0.82rem' }}>{pick(lang, 'Low temperature: almost always the top word. High temperature: more variety.', 'Nízka teplota: takmer vždy najlepšie slovo. Vysoká teplota: väčšia rozmanitosť.')}</p>
    </Frame>
  );
}

// 13 --------------------------------------------------------------------------
const PTS = [[-1.5, -2.7], [-1, -2.3], [-0.5, -0.9], [0, 0.2], [0.5, 1.1], [1, 1.8], [1.5, 3.2], [2, 3.9]];
const loss1d = (w: number) => PTS.reduce((s, [x, y]) => s + (w * x - y) ** 2, 0) / PTS.length;
const grad1d = (w: number) => PTS.reduce((s, [x, y]) => s + 2 * (w * x - y) * x, 0) / PTS.length;
function Train1d() {
  const { lang, num } = useLang();
  const [w, setW] = useState(-0.5);
  const [lr, setLr] = useState(0.2);
  const [hist, setHist] = useState<number[]>([loss1d(-0.5)]);
  const g = grad1d(w);
  const W = 360, H = 180, sx = (v: number) => 20 + ((v + 2) / 4.5) * (W - 40), sy = (v: number) => H - 15 - ((v + 4) / 9) * (H - 30);
  const hmax = Math.max(...hist, 0.1);
  return (
    <Frame title={pick(lang, 'learning one weight', 'učenie jednej váhy')} ev={['simulation']} note={pick(lang, 'Model: prediction = w · x. The dots are the training data; the line is the current model. Loss = mean squared error.', 'Model: predpoveď = w · x. Body sú trénovacie dáta; čiara je aktuálny model. Chyba = priemerná štvorcová odchýlka.')}>
      <div className="grid2">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 420 }} role="img" aria-label={pick(lang, `line with slope ${w.toFixed(2)}`, `priamka so sklonom ${w.toFixed(2)}`)}>
          <line x1={sx(-2)} y1={sy(0)} x2={sx(2.5)} y2={sy(0)} stroke="var(--line)" />
          <line x1={sx(0)} y1={sy(-4)} x2={sx(0)} y2={sy(5)} stroke="var(--line)" />
          {PTS.map(([x, y], i) => <circle key={i} cx={sx(x)} cy={sy(y)} r={4} fill="var(--act)" />)}
          <line x1={sx(-2)} y1={sy(-2 * w)} x2={sx(2.5)} y2={sy(2.5 * w)} stroke="var(--ffn)" strokeWidth={3} />
        </svg>
        <div>
          <p className="mono" style={{ fontSize: '0.88rem' }}>
            w = {num(w, 3)}<br />
            {pick(lang, 'loss', 'chyba')} = {num(loss1d(w), 4)}<br />
            {pick(lang, 'gradient', 'gradient')} = {num(g, 3)} {g > 0 ? pick(lang, '(→ decrease w)', '(→ zmenšiť w)') : pick(lang, '(→ increase w)', '(→ zväčšiť w)')}<br />
            w − {num(lr, 2)} × {num(g, 3)} = {num(w - lr * g, 3)}
          </p>
          <Range label={pick(lang, 'learning rate', 'rýchlosť učenia')} value={lr} min={0.01} max={1.2} step={0.01} onChange={setLr} />
          <div className="row">
            <button className="primary" onClick={() => { const nw = w - lr * g; setW(nw); setHist([...hist, loss1d(nw)].slice(-30)); }}>{pick(lang, 'One training step', 'Jeden krok trénovania')}</button>
            <button onClick={() => { setW(-0.5); setHist([loss1d(-0.5)]); }}>{pick(lang, 'reset', 'od začiatku')}</button>
          </div>
          <svg viewBox="0 0 200 50" style={{ width: '100%', maxWidth: 240, marginTop: 6 }} role="img" aria-label={pick(lang, 'loss history', 'história chyby')}>
            <polyline fill="none" stroke="var(--accent)" strokeWidth={2} points={hist.map((h, i) => `${(i / Math.max(1, hist.length - 1)) * 196 + 2},${48 - (h / hmax) * 44}`).join(' ')} />
          </svg>
          <div className="faint" style={{ fontSize: '0.8rem' }}>{pick(lang, 'loss after each step (try a learning rate above 1)', 'chyba po každom kroku (skúste rýchlosť učenia nad 1)')}</div>
        </div>
      </div>
    </Frame>
  );
}

// 14 --------------------------------------------------------------------------
function MemoryMini() {
  const { lang, num } = useLang();
  const [ctx, setCtx] = useState(8192);
  const rows = [QWEN3_8B, QWEN3_30B_A3B].flatMap((p) => {
    const a = accounting(p);
    return [16, 8, 4].map((bits) => ({ label: `${p.name} · ${bits}-bit`, v: weightBytes(a.total, bits) / GB }));
  });
  const kv = [QWEN3_8B, QWEN3_30B_A3B].map((p) => ({ label: `${p.name} K/V`, v: kvCacheBytes(p, { batch: 1, tokens: ctx, bytesPerScalar: 2 }) / GB }));
  return (
    <Frame title={pick(lang, 'how much memory the numbers need', 'koľko pamäte čísla potrebujú')} ev={['derived']} note={pick(lang, 'Weights = parameters × bits ÷ 8 (ideal payload, without format overhead). K/V cache at BF16 for one conversation.', 'Váhy = parametre × bity ÷ 8 (ideálny objem, bez réžie formátu). K/V cache v BF16 pre jeden rozhovor.')}>
      <HBars max={62} items={rows.map((r) => ({ label: r.label, v: r.v, extra: `${num(r.v, 1)} GB` }))} />
      <Range label={pick(lang, 'conversation length (tokens)', 'dĺžka rozhovoru (tokeny)')} value={ctx} min={1024} max={32768} step={1024} onChange={setCtx} fmt={(v) => num(v)} />
      <HBars max={62} items={kv.map((r) => ({ label: r.label, v: r.v, extra: `${num(r.v, 2)} GB` }))} />
    </Frame>
  );
}

const REGISTRY: Record<WidgetId, () => ReactElement> = {
  'next-token': NextToken, 'real-weight': RealWeight, neuron: Neuron, silu: SiluPlot, matvec: MatVec, tokens: Tokens, rms: Rms,
  attention: AttentionMini, gated: Gated, router: RouterMini, stack: Stack, sampling: SamplingMini, train1d: Train1d, memory: MemoryMini,
  ...REGISTRY2,
};

export function Widget({ id }: { id: WidgetId }) {
  const C = useMemo(() => REGISTRY[id], [id]);
  return <C />;
}

export const WIDGET_IDS = Object.keys(REGISTRY) as WidgetId[];
