import { useState } from 'react';
import { useTrace } from '../data/data';
import { Cite, Evidence, Section } from '../components/ui';
import { useLang } from '../i18n/i18n';
import { QWEN3_0_6B } from '../models/profiles';
import { accounting } from '../models/shapes';

/** Heat colour scale for values in [0, 1]; text is always shown too (colour is never the only cue). */
// Intensity is capped per theme (--heat-cap) so the numbers printed on the cells keep at least 4.5:1 contrast.
const heat = (v: number) => `color-mix(in srgb, var(--attn) calc(var(--heat-cap, 60%) * ${Math.max(0, Math.min(1, v)).toFixed(3)}), var(--bg-2))`;
const signed = (v: number, m: number) => v >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round(Math.min(1, v / m) * 90)}%, var(--bg-2))` : `color-mix(in srgb, var(--danger) ${Math.round(Math.min(1, -v / m) * 90)}%, var(--bg-2))`;

const HOOKS_SK: Record<string, string> = {
  embedding: 'výstup model.model.embed_tokens',
  blocks: 'výstup model.model.layers[i] (reziduálny prúd po bloku)',
  attention: 'druhý výstup model.model.layers[i].self_attn (pravdepodobnosti softmaxu v režime eager) pre bloky [0, 13, 27], hlavy [0, 1, 8, 15]',
  gate: 'SiLU(x @ mlp.gate_proj.weight.T) prepočítané zo zachyteného vstupu MLP, bloky [0, 13, 27], prvých 32 kanálov',
  finalNorm: 'výstup model.model.norm',
};

export default function TracePage() {
  const { lang, num } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const tr = useTrace();
  const [pid, setPid] = useState('raw-capital');
  const [blk, setBlk] = useState(13);
  const [head, setHead] = useState(0);
  const [qpos, setQpos] = useState<number | null>(null);
  const a = accounting(QWEN3_0_6B);
  if (tr.status === 'loading') return <div className="page muted">{T('Loading the measured trace…', 'Načítavam nameraný beh…')}</div>;
  if (tr.status !== 'ok') return <div className="page"><h1>{T('Measured trace', 'Nameraný beh')}</h1><div className="callout warn">{T('The trace file could not be loaded.', 'Súbor s nameraným behom sa nepodarilo načítať.')} {tr.status === 'error' ? tr.error : ''}</div></div>;
  const d = tr.data;
  const p = d.prompts.find((x) => x.id === pid) ?? d.prompts[0];
  const Tn = p.ids.length;
  const att = p.attention.find((x) => x.block === blk && x.head === head) ?? p.attention[0];
  const q = qpos === null ? Tn - 1 : Math.min(qpos, Tn - 1);
  const piece = (s: string) => s.replace(/ /g, '␣').replace(/\n/g, '↵') || '∅';
  const rmsLast = p.residualRms.blocks.map((b) => b[b.length - 1]);
  const maxRms = Math.max(...rmsLast, p.residualRms.embedding[Tn - 1]);
  const feats = p.residualFeaturesLastPos.blocks;
  const fmax = Math.max(...feats.flat().map(Math.abs));
  const W = 560, H = 190, sx = (i: number) => 40 + (i / (rmsLast.length)) * (W - 60), sy = (v: number) => H - 24 - (Math.log10(v) - Math.log10(0.1)) / (Math.log10(maxRms * 1.2) - Math.log10(0.1)) * (H - 40);
  return (
    <div className="page">
      <h1>{T('Measured trace: a real Qwen3 model at work', 'Nameraný beh: skutočný model Qwen3 pri práci')}</h1>
      <div className="badge-row" style={{ marginBottom: 8 }}><Evidence kind="sample" /><Evidence kind="verified" /></div>
      <div className="callout warn">
        {T(`These numbers were really computed by Qwen/Qwen3-0.6B, the smallest dense member of the same Qwen3 family (28 blocks, width 1,024, ${num(a.total)} learned parameters). They are not activations of Qwen3-8B or Qwen3-30B-A3B, which are too large to run here. Everywhere else on this site, activations come from the toy simulator.`,
          `Tieto čísla skutočne vypočítal model Qwen/Qwen3-0.6B, najmenší hustý člen tej istej rodiny Qwen3 (28 blokov, šírka 1 024, ${num(a.total)} naučených parametrov). Nie sú to aktivácie modelov Qwen3-8B ani Qwen3-30B-A3B, ktoré sú na spustenie tu príliš veľké. Všade inde na tejto stránke pochádzajú aktivácie z hračkového simulátora.`)}
      </div>
      <Section title={T('Prompt', 'Zadanie')}>
        <div className="row" role="group" aria-label={T('Prompt', 'Zadanie')}>
          {d.prompts.map((x) => <button key={x.id} className="small" aria-pressed={x.id === p.id} onClick={() => { setPid(x.id); setQpos(null); }}>{x.kind === 'chat' ? T('chat: ', 'rozhovor: ') + x.messages![0].content : x.text}</button>)}
        </div>
        <div className="row" style={{ gap: 4, marginTop: 8 }}>
          {p.pieces.map((s, i) => <button key={i} className={`chip${i === q ? ' sel' : ''}`} onClick={() => setQpos(i)} aria-label={`${T('position', 'pozícia')} ${i}: ${JSON.stringify(s)}, id ${p.ids[i]}`}><span>{piece(s)}</span><span className="faint">{p.ids[i]}</span></button>)}
        </div>
        <h3>{T('What the model predicts next', 'Čo model predpovedá ako ďalšie')}</h3>
        {p.topNext.map((x) => (
          <div key={x.id} className="prob-row" style={{ gridTemplateColumns: '120px minmax(0,1fr) 80px' }}>
            <span className="mono">{piece(x.piece)}</span>
            <span className="track" aria-hidden="true"><span className="fill" style={{ width: `${x.p * 100}%`, display: 'block' }} /></span>
            <span className="mono num" style={{ fontSize: '0.8rem' }}>{num(x.p * 100, 1)} %</span>
          </div>
        ))}
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Probabilities after softmax over all 151,936 vocabulary rows; showing the top 10.', 'Pravdepodobnosti po softmaxe cez všetkých 151 936 riadkov slovníka; zobrazených je 10 najvyšších.')}</p>
      </Section>
      <Section title={T('How large the residual stream gets, block by block', 'Ako narastá reziduálny prúd blok po bloku')} badges={<Evidence kind="sample" />}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 640 }} role="img" aria-label={T('Root-mean-square of the last token’s residual vector after each block (log scale)', 'Stredná kvadratická hodnota vektora posledného tokenu za každým blokom (logaritmická mierka)')}>
          {[0.1, 1, 10, 100].filter((v) => v < maxRms * 1.2).map((v) => <g key={v}><line x1={36} x2={W - 10} y1={sy(v)} y2={sy(v)} stroke="var(--line-soft)" /><text x={32} y={sy(v) + 3} fontSize={9} textAnchor="end" fill="var(--text-3)">{num(v)}</text></g>)}
          <polyline fill="none" stroke="var(--act)" strokeWidth={2.5} points={rmsLast.map((v, i) => `${sx(i + 1)},${sy(v)}`).join(' ')} />
          {rmsLast.map((v, i) => <circle key={i} cx={sx(i + 1)} cy={sy(v)} r={2.5} fill="var(--act)"><title>{`${T('block', 'blok')} ${i}: ${v}`}</title></circle>)}
          <text x={sx(0.3)} y={H - 6} fontSize={9} fill="var(--text-3)">{T('embedding', 'embedding')}</text>
          {[0, 7, 14, 21, 27].map((i) => <text key={i} x={sx(i + 1)} y={H - 6} fontSize={9} textAnchor="middle" fill="var(--text-3)">{i}</text>)}
        </svg>
        <p>{T(`The vector of the last token starts with RMS ${num(p.residualRms.embedding[Tn - 1], 3)} after the embedding lookup and ends with ${num(rmsLast[rmsLast.length - 1], 2)} after block 27. Each block adds to the stream (see the chapter on the residual stream); the final RMSNorm then rescales it (RMS ${num(p.residualRms.finalNorm[Tn - 1], 2)}) before the output head.`, `Vektor posledného tokenu má po vyhľadaní embeddingu RMS ${num(p.residualRms.embedding[Tn - 1], 3)} a za blokom 27 až ${num(rmsLast[rmsLast.length - 1], 2)}. Každý blok k prúdu pripočíta (pozri kapitolu o reziduálnom prúde); záverečná RMSNorm ho potom pred výstupnou hlavou preškáluje (RMS ${num(p.residualRms.finalNorm[Tn - 1], 2)}).`)}</p>
      </Section>
      <Section title={T('Attention: which earlier tokens one head reads', 'Pozornosť: ktoré predchádzajúce tokeny číta jedna hlava')} badges={<Evidence kind="sample" />}>
        <div className="row">
          <label className="field inline">{T('block', 'blok')}<select value={blk} onChange={(e) => setBlk(Number(e.target.value))}>{[...new Set(p.attention.map((x) => x.block))].map((b) => <option key={b} value={b}>{b}</option>)}</select></label>
          <label className="field inline">{T('query head', 'hlava dopytov')}<select value={head} onChange={(e) => setHead(Number(e.target.value))}>{[...new Set(p.attention.map((x) => x.head))].map((h) => <option key={h} value={h}>{h}</option>)}</select></label>
          <span className="faint">{T(`uses K/V head ${att.kvHead} (2 query heads per K/V head)`, `používa hlavu K/V ${att.kvHead} (2 hlavy dopytov na hlavu K/V)`)}</span>
        </div>
        <div className="table-wrap" tabIndex={0} style={{ marginTop: 8 }}>
          <div className="matrix-grid" style={{ gridTemplateColumns: `70px repeat(${Tn}, minmax(30px, 1fr))` }} role="grid" aria-label={T('attention probabilities, rows = query positions, columns = key positions', 'pravdepodobnosti pozornosti, riadky = pozície dopytov, stĺpce = pozície kľúčov')}>
            <div role="row" style={{ display: 'contents' }}>
              <div className="cell hdr" role="columnheader">q \ k</div>
              {p.pieces.map((s, j) => <div key={j} className="cell hdr" role="columnheader" title={s}>{piece(s).slice(0, 4)}</div>)}
            </div>
            {att.probs.map((row, i) => <div key={`r${i}`} role="row" style={{ display: 'contents' }}>{[
              <div key={`h${i}`} role="rowheader" style={{ display: 'contents' }}><button className={`cell hdr${i === q ? ' sel' : ''}`} onClick={() => setQpos(i)} style={{ cursor: 'pointer' }}>{piece(p.pieces[i]).slice(0, 6)}</button></div>,
              ...row.map((v, j) => <div key={`${i}-${j}`} role="gridcell" className={`cell${j > i ? ' na' : ''}`} style={{ background: j > i ? undefined : heat(v) }} title={`${v}`}>{j > i ? '×' : v < 0.005 ? '·' : num(v, 2)}</div>),
            ]}</div>)}
          </div>
        </div>
        <p className="faint" style={{ fontSize: '0.85rem' }}>{T('× = future position, probability exactly 0. In the deeper blocks recorded here (13 and 27), every recorded head puts most of its attention on the very first token for all three prompts, while block 0 spreads it more. This “attention sink” pattern is known in trained transformers and does not mean that token is semantically important. Attention maps alone are not a causal explanation of the answer.', '× = budúca pozícia, pravdepodobnosť presne 0. V hlbších zaznamenaných blokoch (13 a 27) dáva každá zaznamenaná hlava pri všetkých troch zadaniach väčšinu pozornosti úplne prvému tokenu, kým blok 0 ju rozkladá viac. Tento jav („attention sink“) je u natrénovaných transformerov známy a neznamená, že tento token je významovo dôležitý. Samotné mapy pozornosti nie sú príčinným vysvetlením odpovede.')}</p>
      </Section>
      <Section title={T('The first 32 numbers of the last token’s vector, after every block', 'Prvých 32 čísel vektora posledného tokenu za každým blokom')} badges={<Evidence kind="sample" />}>
        <div className="table-wrap" tabIndex={0}>
          <svg viewBox={`0 0 ${32 * 14 + 50} ${feats.length * 11 + 20}`} style={{ width: '100%', maxWidth: 620 }} role="img" aria-label={T('Heatmap of residual features 0–31 across the 28 blocks', 'Teplotná mapa vlastností 0–31 reziduálneho prúdu cez 28 blokov')}>
            {feats.map((row, i) => (
              <g key={i}>
                <text x={44} y={i * 11 + 18} fontSize={8} textAnchor="end" fill="var(--text-3)">{i}</text>
                {row.map((v, j) => <rect key={j} x={50 + j * 14} y={i * 11 + 10} width={13} height={10} fill={signed(v, fmax * 0.4)}><title>{`${T('block', 'blok')} ${i}, ${T('feature', 'vlastnosť')} ${j}: ${v}`}</title></rect>)}
              </g>
            ))}
          </svg>
        </div>
        <p className="faint" style={{ fontSize: '0.85rem' }}>{T('Rows = blocks 0–27, columns = features 0–31 (of 1,024). Blue positive, red negative; hover for the exact value. In the last blocks one of these features reaches about 140–270 while the vector’s RMS stays around 12–21; such “massive activations” in a few features are a known property of trained transformers. No meaning is assigned to individual features.', 'Riadky = bloky 0–27, stĺpce = vlastnosti 0–31 (z 1 024). Modrá kladné, červená záporné; presná hodnota sa zobrazí po prejdení myšou. V posledných blokoch dosiahne jedna z týchto vlastností hodnotu približne 140–270, kým RMS vektora zostáva okolo 12–21; takéto „masívne aktivácie“ v niekoľkých vlastnostiach sú známou vlastnosťou natrénovaných transformerov. Jednotlivým vlastnostiam sa nepripisuje žiadny význam.')}</p>
      </Section>
      <Section title={T('Gated FFN channels: SiLU(gate) for 32 channels, last token', 'Hradlované kanály FFN: SiLU(gate) pre 32 kanálov, posledný token')} badges={<Evidence kind="sample" />}>
        {p.gateLastPos.map((g) => {
          const m = Math.max(...g.silu.map(Math.abs), 1e-9);
          return (
            <div key={g.block} style={{ marginBottom: 8 }}>
              <div className="faint" style={{ fontSize: '0.82rem' }}>{T('block', 'blok')} {g.block}</div>
              <div className="bars" style={{ height: 70 }} aria-label={`${T('block', 'blok')} ${g.block}`}>
                {g.silu.map((v, j) => <div key={j} className="b" title={`${T('channel', 'kanál')} ${j}: ${v}`} style={{ height: `${Math.max(2, (Math.abs(v) / m) * 100)}%`, background: v < 0 ? 'var(--danger)' : 'var(--ffn)' }} />)}
              </div>
            </div>
          );
        })}
        <p className="faint" style={{ fontSize: '0.85rem' }}>{T('Each channel gets its own gate value: some are close to zero (almost closed), others are large (open), and SiLU also allows small negative values (down to −0.28). The gate decides how much of each channel’s “up” value passes (see the chapter on the feed-forward network). Shown for 32 of 3,072 channels per block.', 'Každý kanál má vlastnú hodnotu hradla: niektoré sú blízko nuly (takmer zatvorené), iné veľké (otvorené) a SiLU pripúšťa aj malé záporné hodnoty (až −0,28). Hradlo rozhoduje, koľko z hodnoty „up“ každého kanála prejde (pozri kapitolu o doprednej sieti). Zobrazených je 32 z 3 072 kanálov v bloku.')}</p>
      </Section>
      <Section title={T('Provenance', 'Pôvod údajov')} badges={<Evidence kind="verified" />}>
        <dl className="kv" style={{ fontSize: '0.88rem' }}>
          <dt>{T('model', 'model')}</dt><dd className="mono">{d.model} @ {d.revision}</dd>
          <dt>model.safetensors</dt><dd className="mono">SHA-256 {d.fileSha256['model.safetensors']}</dd>
          <dt>{T('checkpoint file', 'súbor checkpointu')}</dt><dd>{T(`${d.checkpointHeader.tensors} tensors, ${num(d.checkpointHeader.elements)} stored elements (${d.checkpointHeader.dtypes.join(', ')}); it includes lm_head.weight as a byte-identical copy of the tied embedding, so the unique learned parameters are ${num(a.total)}.`, `${d.checkpointHeader.tensors} tenzorov, ${num(d.checkpointHeader.elements)} uložených prvkov (${d.checkpointHeader.dtypes.join(', ')}); obsahuje aj lm_head.weight ako bajtovo zhodnú kópiu viazaného embeddingu, takže jedinečných naučených parametrov je ${num(a.total)}.`)}</dd>
          <dt>{T('software', 'softvér')}</dt><dd className="mono">torch {d.runtime.torch}, transformers {d.runtime.transformers}, Python {d.runtime.python}</dd>
          <dt>{T('precision', 'presnosť')}</dt><dd>{T(`${d.runtime.precision}; ${d.runtime.attnImplementation}`, 'výpočet vo float32; váhy uložené v checkpointe ako BF16 a pri načítaní presne prevedené na float32; pozornosť v režime eager (potrebná na vrátenie pravdepodobností pozornosti)')}</dd>
          <dt>{T('machine', 'počítač')}</dt><dd className="mono">{d.runtime.platform}, {d.runtime.threads} {T('threads', 'vlákien')}; {T('forward passes took', 'priechody modelom trvali')} {num(d.runtime.forwardSeconds, 2)} s</dd>
          <dt>{T('hooks', 'háčiky')}</dt><dd>{Object.entries(d.hooks).map(([k, v]) => <div key={k}><code>{k}</code>: {lang === 'sk' ? HOOKS_SK[k] ?? v : v}</div>)}</dd>
          <dt>{T('prompt text', 'text zadania')}</dt><dd><pre className="mono" style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{p.renderedText}</pre></dd>
          <dt>{T('recorded', 'zaznamenané')}</dt><dd>{d.recorded} · {T(d.rounding, 'čísla zaokrúhlené na 6 platných číslic')} · {T(d.license, 'Apache-2.0 (licencia repozitára modelu)')}</dd>
        </dl>
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Reproduce with scripts/record_traces.py (see docs/checkpoint-extraction.md).', 'Zopakovať sa dá skriptom scripts/record_traces.py (pozri docs/checkpoint-extraction.md).')}<Cite ids={['q06-trace', 'q06-config', 'q06-card']} /></p>
      </Section>
    </div>
  );
}
