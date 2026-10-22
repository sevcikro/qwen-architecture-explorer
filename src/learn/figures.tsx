import { useId, type ReactNode } from 'react';
import { useLang } from '../i18n/i18n';

// Explanatory diagrams for the course. Pure SVG, theme colors via CSS variables, every label in EN and SK.
// Mobile first: viewBox width 360, so text stays readable on a phone.

export type FigureId =
  | 'loop' | 'neuron' | 'matvec' | 'vectors' | 'softmax' | 'tokenize' | 'embedding' | 'residual' | 'rmsnorm'
  | 'qkv' | 'mask' | 'gqa' | 'rope' | 'swiglu' | 'moe' | 'block' | 'model' | 'decode' | 'journey'
  | 'train-loop' | 'graph' | 'pipeline' | 'serving' | 'quant' | 'safetensors' | 'overview';

type TT = (en: string, sk: string) => string;
const C = { w: 'var(--weight)', a: 'var(--attn)', f: 'var(--ffn)', act: 'var(--act)', n: 'var(--norm)', r: 'var(--router)', s: 'var(--state)', t: 'var(--text)', t2: 'var(--text-2)', t3: 'var(--text-3)', line: 'var(--line)', p: 'var(--panel)', p2: 'var(--panel-2)', bg: 'var(--bg-2)' };
const fillOf = (c: string, pct = 18) => `color-mix(in srgb, ${c} ${pct}%, var(--panel))`;

function Svg({ h, label, children }: { h: number; label: string; children: (mk: string) => ReactNode }) {
  const id = useId().replace(/:/g, '');
  const mk = `ar${id}`;
  return (
    <svg viewBox={`0 0 360 ${h}`} role="img" aria-label={label}>
      <defs>
        <marker id={mk} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--text-2)" /></marker>
      </defs>
      {children(mk)}
    </svg>
  );
}
function Box({ x, y, w, h, c = C.t3, label, sub, dash, r = 8, fs }: { x: number; y: number; w: number; h: number; c?: string; label?: string; sub?: string; dash?: boolean; r?: number; fs?: number }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={r} fill={fillOf(c)} stroke={c} strokeWidth={1.6} strokeDasharray={dash ? '5 4' : undefined} />
      {label && <text x={x + w / 2} y={y + h / 2 + (sub ? -3 : 4)} textAnchor="middle" className="fx-label" style={fs ? { fontSize: fs } : undefined}>{label}</text>}
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" className="fx-sub">{sub}</text>}
    </g>
  );
}
const Arr = ({ x1, y1, x2, y2, mk, c = 'var(--text-2)', dash }: { x1: number; y1: number; x2: number; y2: number; mk: string; c?: string; dash?: boolean }) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={1.6} markerEnd={`url(#${mk})`} strokeDasharray={dash ? '4 3' : undefined} />
);
const Txt = ({ x, y, children, cls = 'fx-sub', a = 'middle', c }: { x: number; y: number; children: ReactNode; cls?: string; a?: 'start' | 'middle' | 'end'; c?: string }) => (
  <text x={x} y={y} textAnchor={a} className={cls} style={c ? { fill: c } : undefined}>{children}</text>
);
const Plus = ({ x, y }: { x: number; y: number }) => (
  <g><circle cx={x} cy={y} r={11} fill={C.p} stroke={C.n} strokeWidth={1.8} /><text x={x} y={y + 5} textAnchor="middle" className="fx-label" style={{ fill: C.n, fontSize: 16 }}>+</text></g>
);

// ---------------------------------------------------------------------------------------------
const FIGS: Record<FigureId, (T: TT) => { h: number; label: string; body: (mk: string) => ReactNode }> = {
  loop: (T) => ({
    h: 330, label: T('The generation loop: text to tokens, the model scores every possible next token, one is picked and appended, repeat.', 'Slučka generovania: text na tokeny, model ohodnotí každý možný ďalší token, jeden sa vyberie a pripojí, opakovať.'),
    body: (mk) => (<>
      <Box x={60} y={10} w={240} h={40} c={C.t3} label="“The cat sat on the”" sub={T('text so far', 'doterajší text (po anglicky)')} />
      <Arr x1={180} y1={50} x2={180} y2={70} mk={mk} />
      <Box x={60} y={72} w={240} h={36} c={C.act} label="[785, 8251, 7578, 389, 279]" sub={T('token ids', 'ID tokenov')} fs={12} />
      <Arr x1={180} y1={108} x2={180} y2={128} mk={mk} />
      <Box x={60} y={130} w={240} h={46} c={C.w} label={T('The model (Qwen3-0.6B)', 'Model (Qwen3-0.6B)')} sub={T('596 million fixed weights, 28 blocks', '596 miliónov pevných váh, 28 blokov')} />
      <Arr x1={180} y1={176} x2={180} y2={196} mk={mk} />
      <Box x={60} y={198} w={240} h={70} c={C.a} />
      <Txt x={180} y={213} cls="fx-label">{T('probability of every next token', 'pravdepodobnosť ďalšieho tokenu')}</Txt>
      {[['edge', 0.084], ['window', 0.066], ['floor', 0.065], ['chair', 0.054]].map(([w, p], i) => (
        <g key={i}><text x={76} y={233 + i * 9} className="fx-small fx-mono">{w}</text><rect x={130} y={226 + i * 9} width={(p as number) * 1500} height={7} rx={2} fill={C.a} /><text x={136 + (p as number) * 1500} y={233 + i * 9} className="fx-small">{(p as number).toFixed(3)}</text></g>
      ))}
      <Arr x1={180} y1={268} x2={180} y2={288} mk={mk} />
      <Box x={60} y={290} w={240} h={32} c={C.r} label={T('pick one, e.g. “ window” → append', 'vybrať, napr. „ window“, a pripojiť')} />
      <path d="M300,306 C350,306 350,30 302,30" fill="none" stroke="var(--text-2)" strokeWidth={1.6} strokeDasharray="5 4" markerEnd={`url(#${mk})`} />
      <text x={352} y={170} textAnchor="middle" className="fx-small" transform="rotate(90 352 170)">{T('repeat', 'znova')}</text>
    </>),
  }),
  neuron: (T) => ({
    h: 220, label: T('One neuron: each input is multiplied by its weight, the products and a bias are summed, then an activation function is applied.', 'Jeden neurón: každý vstup sa vynásobí svojou váhou, súčiny a bias sa sčítajú a potom sa použije aktivačná funkcia.'),
    body: (mk) => (<>
      {[['x₁ = 0.5', 'w₁ = 0.8'], ['x₂ = −1.0', 'w₂ = 0.1'], ['x₃ = 2.0', 'w₃ = −0.4']].map(([x, w], i) => (
        <g key={i}>
          <Box x={8} y={20 + i * 62} w={78} h={34} c={C.act} label={x} fs={12} />
          <Arr x1={86} y1={37 + i * 62} x2={186} y2={108} mk={mk} c={C.w} />
          <text x={[116, 116, 124][i]} y={[60, 122, 164][i]} className="fx-sub" style={{ fill: C.w }}>{w}</text>
        </g>
      ))}
      <circle cx={208} cy={110} r={24} fill={fillOf(C.w)} stroke={C.w} strokeWidth={1.8} />
      <text x={208} y={117} textAnchor="middle" className="fx-label" style={{ fontSize: 20 }}>Σ</text>
      <Txt x={208} y={152} cls="fx-small">+ b = 0.2</Txt>
      <Arr x1={232} y1={110} x2={258} y2={110} mk={mk} />
      <Box x={260} y={88} w={56} h={44} c={C.act} label="f(z)" sub="SiLU" />
      <Arr x1={316} y1={110} x2={350} y2={110} mk={mk} />
      <Txt x={208} y={190} cls="fx-sub">{T('z = Σ wᵢxᵢ + b = −0.3', 'z = Σ wᵢxᵢ + b = −0,3')}</Txt>
      <Txt x={208} y={208} cls="fx-sub">{T('output = SiLU(−0.3) = −0.128', 'výstup = SiLU(−0,3) = −0,128')}</Txt>
    </>),
  }),
  matvec: (T) => ({
    h: 210, label: T('Matrix times vector: every row of W is one neuron; row i dotted with x gives output y_i.', 'Matica krát vektor: každý riadok W je jeden neurón; skalárny súčin riadku i s x dá výstup y_i.'),
    body: (mk) => (<>
      <Txt x={80} y={18} cls="fx-label">W [4 × 3]</Txt>
      {Array.from({ length: 4 }, (_, r) => Array.from({ length: 3 }, (_, c) => (
        <rect key={`${r}${c}`} x={30 + c * 34} y={28 + r * 34} width={32} height={32} rx={3} fill={r === 1 ? fillOf(C.w, 45) : fillOf(C.w, 14)} stroke={C.w} />
      )))}
      <Txt x={164} y={100} cls="fx-label">×</Txt>
      <Txt x={200} y={18} cls="fx-label">x [3]</Txt>
      {Array.from({ length: 3 }, (_, i) => <rect key={i} x={186} y={45 + i * 34} width={30} height={32} rx={3} fill={fillOf(C.act, 30)} stroke={C.act} />)}
      <Txt x={240} y={100} cls="fx-label">=</Txt>
      <Txt x={290} y={18} cls="fx-label">y [4]</Txt>
      {Array.from({ length: 4 }, (_, i) => <rect key={i} x={275} y={28 + i * 34} width={30} height={32} rx={3} fill={i === 1 ? fillOf(C.act, 55) : fillOf(C.act, 16)} stroke={C.act} />)}
      <Arr x1={132} y1={79} x2={272} y2={79} mk={mk} c={C.w} dash />
      <Txt x={180} y={185} cls="fx-sub">{T('y₁ = W[1,0]·x₀ + W[1,1]·x₁ + W[1,2]·x₂', 'y₁ = W[1,0]·x₀ + W[1,1]·x₁ + W[1,2]·x₂')}</Txt>
      <Txt x={180} y={202} cls="fx-small">{T('12 weights = 4 neurons × 3 inputs', '12 váh = 4 neuróny × 3 vstupy')}</Txt>
    </>),
  }),
  vectors: (T) => ({
    h: 230, label: T('Vectors as arrows: similar meanings point in similar directions, so their dot product and cosine are large.', 'Vektory ako šípky: podobné významy ukazujú podobným smerom, preto majú veľký skalárny súčin a kosínus.'),
    body: (mk) => (<>
      <line x1={40} y1={200} x2={340} y2={200} stroke={C.line} /><line x1={40} y1={200} x2={40} y2={14} stroke={C.line} />
      <Arr x1={40} y1={200} x2={250} y2={60} mk={mk} c={C.a} /><Txt x={258} y={56} cls="fx-label" a="start">{T('cat', 'mačka')}</Txt>
      <Arr x1={40} y1={200} x2={230} y2={40} mk={mk} c={C.a} /><Txt x={226} y={32} cls="fx-label" a="end">{T('dog', 'pes')}</Txt>
      <Arr x1={40} y1={200} x2={320} y2={178} mk={mk} c={C.r} /><Txt x={318} y={170} cls="fx-label" a="end">{T('car', 'auto')}</Txt>
      <Txt x={150} y={150} cls="fx-small" a="start">{T('small angle → cos ≈ 0.99', 'malý uhol → cos ≈ 0,99')}</Txt>
      <Txt x={200} y={196} cls="fx-small" a="start">{T('large angle → cos ≈ 0.3', 'veľký uhol → cos ≈ 0,3')}</Txt>
      <Txt x={190} y={222} cls="fx-sub">a · b = |a| |b| cos θ</Txt>
    </>),
  }),
  softmax: (T) => ({
    h: 220, label: T('Softmax: raw scores (logits) are exponentiated and divided by their sum, giving probabilities that add up to 1.', 'Softmax: surové skóre (logity) sa umocnia a vydelia ich súčtom, čím vzniknú pravdepodobnosti so súčtom 1.'),
    body: (mk) => {
      const L = [4, 2.5, 2, -1], P = [0.7325, 0.1634, 0.0991, 0.0049], N = ['Paris', 'London', 'Rome', 'banana'];
      return (<>
        <Txt x={90} y={16} cls="fx-label">{T('logits (any sign)', 'logity (ľub. znamienko)')}</Txt>
        <Txt x={270} y={16} cls="fx-label">{T('probabilities', 'pravdepodobnosti')}</Txt>
        <line x1={90} y1={24} x2={90} y2={186} stroke={C.line} />
        {L.map((v, i) => (
          <g key={i}>
            <text x={8} y={44 + i * 40} className="fx-small fx-mono">{N[i]}</text>
            <rect x={v >= 0 ? 90 : 90 + v * 18} y={32 + i * 40} width={Math.abs(v) * 18} height={18} rx={3} fill={v >= 0 ? C.a : C.r} />
            <text x={v >= 0 ? 94 + v * 18 : 94} y={46 + i * 40} className="fx-small">{String(v)}</text>
            <rect x={210} y={32 + i * 40} width={P[i] * 140} height={18} rx={3} fill={C.act} />
            <text x={214 + P[i] * 140} y={46 + i * 40} className="fx-small">{P[i].toFixed(3)}</text>
          </g>
        ))}
        <Arr x1={168} y1={100} x2={204} y2={100} mk={mk} />
        <Txt x={186} y={92} cls="fx-small">eˣ / Σ</Txt>
        <Txt x={180} y={208} cls="fx-sub">{T('p_i = exp(z_i) / Σ_j exp(z_j);  Σ p = 1', 'p_i = exp(z_i) / Σ_j exp(z_j);  Σ p = 1')}</Txt>
      </>);
    },
  }),
  tokenize: (T) => ({
    h: 230, label: T('Tokenization: text is split into pieces from a fixed vocabulary, each piece has an id, and each id selects one row of the embedding table.', 'Tokenizácia: text sa rozdelí na kúsky z pevného slovníka, každý kúsok má ID a každé ID vyberie jeden riadok tabuľky embeddingov.'),
    body: (mk) => {
      const pieces = [['The', 785], [' capital', 6722], [' of', 315], [' France', 9625]] as const;
      return (<>
        <Box x={40} y={8} w={280} h={32} c={C.t3} label="“The capital of France”" />
        <Arr x1={180} y1={40} x2={180} y2={58} mk={mk} />
        <Txt x={190} y={54} cls="fx-small" a="start">{T('tokenizer (byte-level BPE)', 'tokenizér (BPE na úrovni bajtov)')}</Txt>
        {pieces.map(([p, id], i) => (
          <g key={i}>
            <Box x={8 + i * 88} y={62} w={80} h={30} c={C.act} label={`“${p.trim() === p ? p : '␣' + p.trim()}”`} fs={12} />
            <Arr x1={48 + i * 88} y1={92} x2={48 + i * 88} y2={108} mk={mk} />
            <Box x={8 + i * 88} y={110} w={80} h={26} c={C.act} label={String(id)} fs={12} />
            <Arr x1={48 + i * 88} y1={136} x2={48 + i * 88} y2={156} mk={mk} c={C.w} />
            {Array.from({ length: 6 }, (_, k) => <rect key={k} x={12 + i * 88 + k * 12} y={160} width={11} height={22} fill={fillOf(C.w, 20 + ((id * (k + 3)) % 60))} stroke={C.w} strokeWidth={0.6} />)}
          </g>
        ))}
        <Txt x={180} y={202} cls="fx-sub">{T('each id → one row of the embedding table (4,096 numbers in Qwen3-8B)', 'každé ID → jeden riadok tabuľky embeddingov (4 096 čísel v Qwen3-8B)')}</Txt>
        <Txt x={180} y={220} cls="fx-small">{T('real ids from the Qwen3 tokenizer', 'skutočné ID z tokenizéra Qwen3')}</Txt>
      </>);
    },
  }),
  embedding: (T) => ({
    h: 200, label: T('An embedding lookup is the same as multiplying a one-hot vector by the embedding matrix: it selects one row.', 'Vyhľadanie embeddingu je to isté ako násobenie one-hot vektora maticou embeddingov: vyberie jeden riadok.'),
    body: (mk) => (<>
      <Txt x={60} y={16} cls="fx-label">one-hot</Txt>
      {Array.from({ length: 8 }, (_, i) => <rect key={i} x={50} y={24 + i * 18} width={20} height={16} rx={2} fill={i === 5 ? C.act : fillOf(C.act, 8)} stroke={C.act} strokeWidth={0.8} />)}
      {Array.from({ length: 8 }, (_, i) => <text key={i} x={76} y={36 + i * 18} className="fx-small">{i === 5 ? '1' : '0'}</text>)}
      <Txt x={110} y={100} cls="fx-label">×</Txt>
      <Txt x={200} y={16} cls="fx-label">{T('E [vocab × d]', 'E [slovník × d]')}</Txt>
      {Array.from({ length: 8 }, (_, r) => Array.from({ length: 6 }, (_, c) => <rect key={`${r}${c}`} x={140 + c * 20} y={24 + r * 18} width={19} height={16} fill={r === 5 ? fillOf(C.w, 60) : fillOf(C.w, 12)} stroke={C.w} strokeWidth={0.6} />))}
      <Arr x1={262} y1={122} x2={292} y2={122} mk={mk} c={C.w} />
      {Array.from({ length: 6 }, (_, c) => <rect key={c} x={296} y={50 + c * 18} width={40} height={16} fill={fillOf(C.w, 60)} stroke={C.w} strokeWidth={0.6} />)}
      <Txt x={316} y={40} cls="fx-sub">{T('row 5', 'riadok 5')}</Txt>
      <Txt x={180} y={190} cls="fx-sub">{T('in practice: just read row number token_id', 'v praxi: jednoducho sa prečíta riadok číslo token_id')}</Txt>
    </>),
  }),
  residual: (T) => ({
    h: 250, label: T('The residual stream: a vector flows upward; each sub-layer reads a normalized copy and adds its result back.', 'Reziduálny prúd: vektor tečie nahor; každá podvrstva číta normalizovanú kópiu a svoj výsledok k nemu pripočíta.'),
    body: (mk) => (<>
      <line x1={90} y1={238} x2={90} y2={62} stroke={C.n} strokeWidth={5} opacity={0.55} markerEnd={`url(#${mk})`} />
      <Txt x={90} y={248} cls="fx-sub">{T('embedding', 'embedding')}</Txt>
      {[[180, T('attention', 'pozornosť'), C.a], [100, T('feed-forward', 'dopredná sieť'), C.f]].map(([y, l, c], i) => (
        <g key={i}>
          <Arr x1={90} y1={(y as number) + 36} x2={148} y2={(y as number) + 36} mk={mk} c={C.t3} />
          <Box x={150} y={(y as number) + 22} w={70} h={28} c={C.n} label="RMSNorm" fs={11} />
          <Arr x1={220} y1={(y as number) + 36} x2={238} y2={(y as number) + 36} mk={mk} />
          <Box x={240} y={(y as number) + 14} w={110} h={44} c={c as string} label={l as string} />
          <path d={`M295,${(y as number) + 14} L295,${(y as number) - 4} L104,${(y as number) - 4}`} fill="none" stroke={c as string} strokeWidth={1.6} markerEnd={`url(#${mk})`} />
          <Plus x={90} y={(y as number) - 4} />
        </g>
      ))}
      <Txt x={200} y={20} cls="fx-sub">{T('x ← x + Sublayer(Norm(x))', 'x ← x + Podvrstva(Norm(x))')}</Txt>
      <Txt x={200} y={36} cls="fx-small">{T('the additions have no weights;', 'sčítania nemajú váhy;')}</Txt>
      <Txt x={200} y={50} cls="fx-small">{T('the stream keeps its width', 'prúd si drží šírku')}</Txt>
    </>),
  }),
  rmsnorm: (T) => ({
    h: 190, label: T('RMSNorm divides a vector by its root-mean-square and multiplies by learned gains, so its typical size becomes about 1.', 'RMSNorm vydelí vektor jeho strednou kvadratickou hodnotou a vynásobí naučenými ziskami, takže jeho typická veľkosť bude okolo 1.'),
    body: (mk) => {
      const x = [12, -3, 0.5, 40], y = [0.5732, -0.1433, 0.0239, 1.9106];
      return (<>
        <Txt x={90} y={16} cls="fx-label">x</Txt><Txt x={270} y={16} cls="fx-label">RMSNorm(x)</Txt>
        <line x1={20} y1={110} x2={170} y2={110} stroke={C.line} /><line x1={200} y1={110} x2={350} y2={110} stroke={C.line} />
        {x.map((v, i) => <rect key={i} x={30 + i * 34} y={v >= 0 ? 110 - v * 2 : 110} width={26} height={Math.abs(v) * 2} fill={C.act} />)}
        {y.map((v, i) => <rect key={i} x={210 + i * 34} y={v >= 0 ? 110 - v * 40 : 110} width={26} height={Math.abs(v) * 40} fill={C.n} />)}
        <Arr x1={172} y1={70} x2={198} y2={70} mk={mk} />
        <Txt x={90} y={140} cls="fx-small">rms = 20.94</Txt><Txt x={270} y={140} cls="fx-small">rms = 1.00</Txt>
        <Txt x={180} y={168} cls="fx-sub">{T('y = x / √(mean(x²) + ε) · g', 'y = x / √(priemer(x²) + ε) · g')}</Txt>
        <Txt x={180} y={184} cls="fx-small">{T('g = learned gain per coordinate (1,024 numbers in Qwen3-0.6B)', 'g = naučený zisk pre každú súradnicu (1 024 čísel v Qwen3-0.6B)')}</Txt>
      </>);
    },
  }),
  qkv: (T) => ({
    h: 300, label: T('Attention: each token makes a query, key and value; queries are compared with keys of earlier tokens, softmax gives weights, and the output is the weighted average of values.', 'Pozornosť: každý token vytvorí dopyt, kľúč a hodnotu; dopyty sa porovnajú s kľúčmi skorších tokenov, softmax dá váhy a výstup je vážený priemer hodnôt.'),
    body: (mk) => (<>
      <Box x={120} y={6} w={120} h={30} c={C.act} label={T('token vectors X', 'vektory tokenov X')} fs={12} />
      {[['Q = X W_Qᵀ', 62, T('query: what I look for', 'dopyt: čo hľadám?')], ['K = X W_Kᵀ', 180, T('key: what do I offer?', 'kľúč: čo ponúkam?')], ['V = X W_Vᵀ', 298, T('value: what I pass on', 'hodnota: čo odovzdám')]].map(([l, x, s], i) => (
        <g key={i}>
          <Arr x1={180} y1={36} x2={x as number} y2={60} mk={mk} c={C.w} />
          <Box x={(x as number) - 52} y={62} w={104} h={30} c={C.a} label={l as string} fs={12} />
          <Txt x={x as number} y={106} cls="fx-small">{s as string}</Txt>
        </g>
      ))}
      <Arr x1={62} y1={112} x2={100} y2={140} mk={mk} /><Arr x1={180} y1={112} x2={130} y2={140} mk={mk} />
      <Box x={40} y={142} w={150} h={36} c={C.a} label="S = Q Kᵀ / √d_h" sub={T('score of every pair', 'skóre každej dvojice')} fs={12} />
      <Arr x1={115} y1={178} x2={115} y2={196} mk={mk} />
      <Box x={40} y={198} w={150} h={36} c={C.act} label={T('softmax (masked)', 'softmax (s maskou)')} sub={T('rows sum to 1', 'riadky dávajú 1')} fs={12} />
      <Arr x1={190} y1={216} x2={250} y2={250} mk={mk} /><Arr x1={298} y1={112} x2={300} y2={244} mk={mk} />
      <Box x={210} y={246} w={140} h={44} c={C.a} label="A · V" sub={T('weighted average of values', 'vážený priemer hodnôt')} fs={12} />
      <Txt x={110} y={262} cls="fx-sub">{T('then W_O mixes the heads', 'potom W_O zmieša hlavy')}</Txt>
    </>),
  }),
  mask: (T) => ({
    h: 230, label: T('The causal mask: token i may only attend to tokens 0..i, so the score matrix is lower-triangular.', 'Kauzálna maska: token i smie venovať pozornosť len tokenom 0..i, preto je matica skóre dolná trojuholníková.'),
    body: () => {
      const tk = ['The', 'cat', 'sat', 'on', 'the'];
      const A = [[1, 0, 0, 0, 0], [0.4, 0.6, 0, 0, 0], [0.2, 0.5, 0.3, 0, 0], [0.1, 0.2, 0.4, 0.3, 0], [0.3, 0.1, 0.2, 0.1, 0.3]];
      return (<>
        <Txt x={200} y={14} cls="fx-sub">{T('keys (earlier tokens) →', 'kľúče (skoršie tokeny) →')}</Txt>
        {tk.map((t, j) => <text key={j} x={130 + j * 40} y={34} textAnchor="middle" className="fx-small fx-mono">{t}</text>)}
        {tk.map((t, i) => <text key={i} x={104} y={60 + i * 34} textAnchor="end" className="fx-small fx-mono">{t}</text>)}
        {A.map((row, i) => row.map((v, j) => (
          <g key={`${i}${j}`}>
            <rect x={112 + j * 40} y={42 + i * 34} width={36} height={30} rx={3} fill={j > i ? 'var(--bg-2)' : `color-mix(in srgb, var(--attn) ${Math.round(12 + v * 48)}%, var(--panel))`} stroke={j > i ? C.line : C.a} strokeDasharray={j > i ? '3 3' : undefined} />
            <text x={130 + j * 40} y={61 + i * 34} textAnchor="middle" className="fx-sub" style={{ fill: 'var(--text)' }}>{j > i ? '—' : (T('0.0', '0,0') === '0,0' ? v.toFixed(1).replace('.', ',') : v.toFixed(1))}</text>
          </g>
        )))}
        <text x={20} y={120} className="fx-sub" transform="rotate(-90 20 120)" textAnchor="middle">{T('queries ↓', 'dopyty ↓')}</text>
        <Txt x={200} y={222} cls="fx-small">{T('grey = future: score set to −∞, probability exactly 0 (illustrative numbers)', 'sivé = budúcnosť: skóre −∞, pravdepodobnosť presne 0 (ilustračné čísla)')}</Txt>
      </>);
    },
  }),
  gqa: (T) => ({
    h: 200, label: T('Grouped-query attention in Qwen3-8B: 32 query heads share 8 key/value heads, 4 queries per K/V head.', 'Pozornosť so zoskupenými dopytmi v Qwen3-8B: 32 hláv dopytov zdieľa 8 hláv kľúčov/hodnôt, 4 dopyty na jednu hlavu K/V.'),
    body: (mk) => (<>
      <Txt x={180} y={14} cls="fx-sub">{T('32 query heads', '32 hláv dopytov')}</Txt>
      {Array.from({ length: 32 }, (_, i) => <rect key={i} x={8 + i * 10.8} y={22} width={9} height={26} rx={2} fill={`color-mix(in srgb, var(--attn) ${30 + (Math.floor(i / 4) % 2) * 35}%, var(--panel))`} stroke={C.a} strokeWidth={0.7} />)}
      {Array.from({ length: 8 }, (_, g) => <Arr key={g} x1={8 + g * 43.2 + 21} y1={50} x2={30 + g * 43} y2={104} mk={mk} c={C.t3} />)}
      {Array.from({ length: 8 }, (_, g) => <Box key={g} x={12 + g * 43} y={106} w={38} h={34} c={C.s} label={`K/V ${g}`} fs={10} r={4} />)}
      <Txt x={180} y={162} cls="fx-sub">{T('K/V cache stores only 8 heads → 4× smaller than with 32', 'cache K/V ukladá len 8 hláv → 4× menšia než pri 32')}</Txt>
      <Txt x={180} y={180} cls="fx-small">{T('query head h uses K/V head ⌊h / 4⌋', 'hlava dopytu h používa hlavu K/V ⌊h / 4⌋')}</Txt>
    </>),
  }),
  rope: (T) => ({
    h: 230, label: T('RoPE rotates each pair of query/key coordinates by an angle proportional to the position, so the score depends on the distance between tokens.', 'RoPE otáča každý pár súradníc dopytu/kľúča o uhol úmerný pozícii, takže skóre závisí od vzdialenosti medzi tokenmi.'),
    body: (mk) => {
      const cx = 110, cy = 112, R = 80;
      return (<>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={C.line} strokeDasharray="4 4" />
        {[0, 1, 2, 3].map((m) => {
          const a = -(0.15 + m * 0.42);
          return <g key={m}><Arr x1={cx} y1={cy} x2={cx + R * Math.cos(a)} y2={cy + R * Math.sin(a)} mk={mk} c={`color-mix(in srgb, var(--attn) ${40 + m * 20}%, var(--text-3))`} /><text x={cx + (R + 14) * Math.cos(a)} y={cy + (R + 14) * Math.sin(a) + 4} textAnchor="middle" className="fx-small">{T(`pos ${m}`, `poz. ${m}`)}</text></g>;
        })}
        <Txt x={cx} y={cy + R + 26} cls="fx-small">{T('same vector, rotated by m·θ', 'ten istý vektor otočený o m·θ')}</Txt>
        <Txt x={280} y={70} cls="fx-label">q_m · k_n</Txt>
        <Txt x={280} y={90} cls="fx-sub">{T('depends only on m − n', 'závisí len od m − n')}</Txt>
        <Txt x={280} y={126} cls="fx-small">{T('fast pairs: nearby order', 'rýchle páry: blízke poradie')}</Txt>
        <Txt x={280} y={142} cls="fx-small">{T('slow pairs: long range', 'pomalé páry: veľké vzdialenosti')}</Txt>
        <Txt x={280} y={170} cls="fx-small">θ_i = 1 000 000^(−2i/d)</Txt>
        <Txt x={280} y={186} cls="fx-small">{T('no learned parameters', 'bez naučených parametrov')}</Txt>
      </>);
    },
  }),
  swiglu: (T) => ({
    h: 250, label: T('The SwiGLU feed-forward network: two parallel projections (gate and up), SiLU on the gate, elementwise product, then the down projection.', 'Dopredná sieť SwiGLU: dve paralelné projekcie (gate a up), SiLU na gate, súčin po prvkoch a potom projekcia down.'),
    body: (mk) => (<>
      <Box x={130} y={6} w={100} h={30} c={C.n} label={T('x (d = 4096)', 'x (d = 4096)')} fs={12} />
      <Arr x1={160} y1={36} x2={90} y2={62} mk={mk} /><Arr x1={200} y1={36} x2={270} y2={62} mk={mk} />
      <Box x={30} y={64} w={120} h={36} c={C.f} label="gate_proj" sub="[12288 × 4096]" fs={12} />
      <Box x={210} y={64} w={120} h={36} c={C.f} label="up_proj" sub="[12288 × 4096]" fs={12} />
      <Arr x1={90} y1={100} x2={90} y2={118} mk={mk} />
      <Box x={50} y={120} w={80} h={30} c={C.act} label="SiLU" fs={12} />
      <Arr x1={90} y1={150} x2={168} y2={170} mk={mk} /><Arr x1={270} y1={100} x2={192} y2={170} mk={mk} />
      <circle cx={180} cy={178} r={13} fill={C.p} stroke={C.f} strokeWidth={1.8} /><text x={180} y={183} textAnchor="middle" className="fx-label">⊙</text>
      <Arr x1={180} y1={191} x2={180} y2={204} mk={mk} />
      <Box x={110} y={206} w={140} h={36} c={C.f} label="down_proj" sub="[4096 × 12288]" fs={12} />
      <Txt x={300} y={186} cls="fx-small">{T('12,288 hidden', '12 288 skrytých')}</Txt>
      <Txt x={300} y={200} cls="fx-small">{T('“neurons”', '„neurónov“')}</Txt>
    </>),
  }),
  moe: (T) => ({
    h: 250, label: T('Mixture of experts: a router scores all experts, the top-k are run, and their outputs are mixed with renormalized weights.', 'Zmes expertov: smerovač ohodnotí všetkých expertov, spustí sa top-k a ich výstupy sa zmiešajú s prepočítanými váhami.'),
    body: (mk) => (<>
      <Box x={130} y={6} w={100} h={30} c={C.n} label="x" fs={13} />
      <Arr x1={180} y1={36} x2={180} y2={52} mk={mk} />
      <Box x={110} y={54} w={140} h={32} c={C.r} label={T('router: softmax(x W_rᵀ)', 'smerovač: softmax(x W_rᵀ)')} fs={11} />
      {Array.from({ length: 16 }, (_, e) => {
        const on = e === 2 || e === 9;
        return <g key={e}><rect x={10 + e * 21.5} y={110} width={19} height={44} rx={3} fill={on ? fillOf(C.f, 60) : fillOf(C.f, 8)} stroke={on ? C.f : C.line} strokeWidth={on ? 2 : 1} strokeDasharray={on ? undefined : '3 2'} /><text x={19.5 + e * 21.5} y={170} textAnchor="middle" className="fx-small">{e}</text></g>;
      })}
      <Arr x1={170} y1={86} x2={62} y2={106} mk={mk} c={C.r} /><Arr x1={190} y1={86} x2={213} y2={106} mk={mk} c={C.r} />
      <Txt x={92} y={92} cls="fx-small" a="end" c={C.r}>α = 0.7</Txt><Txt x={262} y={92} cls="fx-small" a="start" c={C.r}>α = 0.3</Txt>
      <Arr x1={62} y1={156} x2={168} y2={196} mk={mk} /><Arr x1={213} y1={156} x2={192} y2={196} mk={mk} />
      <Box x={110} y={198} w={140} h={32} c={C.f} label="y = Σ α_e · Expert_e(x)" fs={11} />
      <Txt x={180} y={246} cls="fx-small">{T('Qwen3-30B-A3B: 128 experts, 8 run per token per block', 'Qwen3-30B-A3B: 128 expertov, na token a blok sa spustí 8')}</Txt>
    </>),
  }),
  block: (T) => ({
    h: 360, label: T('One Qwen3 decoder block: RMSNorm, attention with QK-norm and RoPE, residual add, RMSNorm, SwiGLU feed-forward, residual add.', 'Jeden blok dekodéra Qwen3: RMSNorm, pozornosť s QK-norm a RoPE, reziduálne sčítanie, RMSNorm, dopredná sieť SwiGLU, reziduálne sčítanie.'),
    body: (mk) => (<>
      <line x1={60} y1={350} x2={60} y2={26} stroke={C.n} strokeWidth={5} opacity={0.5} markerEnd={`url(#${mk})`} />
      <Txt x={60} y={358} cls="fx-small">{T('x in', 'x vstup')}</Txt><Txt x={30} y={18} cls="fx-small">{T('x out', 'x výstup')}</Txt>
      <Arr x1={60} y1={320} x2={118} y2={320} mk={mk} c={C.t3} />
      <Box x={120} y={306} w={110} h={28} c={C.n} label="input_layernorm" fs={11} />
      <Arr x1={175} y1={306} x2={175} y2={290} mk={mk} />
      <Box x={110} y={244} w={240} h={44} c={C.a} label={T('Q, K, V projections', 'projekcie Q, K, V')} sub={T('q_norm, k_norm, RoPE on Q and K', 'q_norm, k_norm, RoPE na Q a K')} fs={12} />
      <Arr x1={230} y1={244} x2={230} y2={228} mk={mk} />
      <Box x={110} y={196} w={240} h={30} c={C.a} label={T('causal softmax attention + o_proj', 'kauzálna softmax pozornosť + o_proj')} fs={11} />
      <path d="M230,196 L230,186 L72,186" fill="none" stroke={C.a} strokeWidth={1.6} markerEnd={`url(#${mk})`} /><Plus x={60} y={186} />
      <Arr x1={60} y1={150} x2={118} y2={150} mk={mk} c={C.t3} />
      <Box x={120} y={136} w={150} h={28} c={C.n} label="post_attention_layernorm" fs={10.5} />
      <Arr x1={195} y1={136} x2={195} y2={118} mk={mk} />
      <Box x={110} y={72} w={240} h={44} c={C.f} label="SwiGLU: down(SiLU(gate(x)) ⊙ up(x))" sub={T('or MoE: router + top-k experts', 'alebo MoE: smerovač + top-k expertov')} fs={11} />
      <path d="M230,72 L230,52 L72,52" fill="none" stroke={C.f} strokeWidth={1.6} markerEnd={`url(#${mk})`} /><Plus x={60} y={52} />
      <Txt x={240} y={22} cls="fx-small">{T('repeated 36× (8B) / 48× (30B-A3B),', 'opakuje sa 36× (8B) / 48× (30B-A3B),')}</Txt>
      <Txt x={240} y={36} cls="fx-small">{T('each block with its own weights', 'každý blok s vlastnými váhami')}</Txt>
    </>),
  }),
  model: (T) => ({
    h: 300, label: T('The whole decoder-only model: embedding, a stack of identical-shape blocks, final RMSNorm, output head (lm_head) producing one logit per vocabulary entry.', 'Celý model typu iba dekodér: embedding, stoh blokov s rovnakým tvarom, záverečná RMSNorm a výstupná hlava (lm_head), ktorá vytvorí jeden logit pre každú položku slovníka.'),
    body: (mk) => (<>
      <Box x={70} y={262} w={220} h={30} c={C.act} label={T('token ids [T]', 'ID tokenov [T]')} fs={12} />
      <Arr x1={180} y1={262} x2={180} y2={248} mk={mk} />
      <Box x={70} y={216} w={220} h={30} c={C.w} label={T('embed_tokens → [T, 4096]', 'embed_tokens → [T, 4096]')} fs={12} />
      <Arr x1={180} y1={216} x2={180} y2={202} mk={mk} />
      {[0, 1, 2].map((i) => <rect key={i} x={70 + i * 6} y={118 - i * 6} width={220} height={80} rx={8} fill={fillOf(C.a, 10 + i * 6)} stroke={C.a} />)}
      <Txt x={192} y={140} cls="fx-label">{T('36 decoder blocks', '36 blokov dekodéra')}</Txt>
      <Txt x={192} y={158} cls="fx-sub">{T('attention + FFN, residual stream', 'pozornosť + FFN, reziduálny prúd')}</Txt>
      <Txt x={192} y={174} cls="fx-small">{T('shape stays [T, 4096]', 'tvar zostáva [T, 4096]')}</Txt>
      <Arr x1={180} y1={106} x2={180} y2={92} mk={mk} />
      <Box x={70} y={62} w={220} h={28} c={C.n} label={T('final RMSNorm', 'záverečná RMSNorm')} fs={12} />
      <Arr x1={180} y1={62} x2={180} y2={48} mk={mk} />
      <Box x={70} y={10} w={220} h={36} c={C.w} label={T('lm_head → logits [T, 151936]', 'lm_head → logity [T, 151936]')} sub={T('only the last row is needed to predict', 'na predpoveď treba len posledný riadok')} fs={12} />
    </>),
  }),
  decode: (T) => ({
    h: 262, label: T('Prefill processes the whole prompt at once and fills the K/V cache; each decode step processes one new token and appends one K/V entry per block.', 'Predvyplnenie spracuje celé zadanie naraz a naplní cache K/V; každý krok generovania spracuje jeden nový token a pripojí jednu položku K/V v každom bloku.'),
    body: () => (<>
      <Txt x={8} y={16} cls="fx-label" a="start">{T('prefill (all prompt tokens in parallel)', 'predvyplnenie (všetky tokeny zadania paralelne)')}</Txt>
      {Array.from({ length: 5 }, (_, i) => <rect key={i} x={8 + i * 30} y={26} width={26} height={26} rx={4} fill={fillOf(C.act, 40)} stroke={C.act} />)}
      <Txt x={8} y={82} cls="fx-label" a="start">{T('decode (one token per step)', 'generovanie (jeden token na krok)')}</Txt>
      {[0, 1, 2].map((s) => (
        <g key={s}>
          <text x={8} y={102 + s * 44} className="fx-small">{T(`step ${s + 1}: reads ${5 + s} cached K/V + itself`, `krok ${s + 1}: číta ${5 + s} K/V z cache + seba`)}</text>
          {Array.from({ length: 5 + s }, (_, i) => <rect key={i} x={8 + i * 30} y={106 + s * 44} width={26} height={26} rx={4} fill={fillOf(C.s, 25)} stroke={C.s} />)}
          <rect x={8 + (5 + s) * 30} y={106 + s * 44} width={26} height={26} rx={4} fill={fillOf(C.r, 50)} stroke={C.r} strokeWidth={2} />
        </g>
      ))}
      <rect x={8} y={228} width={14} height={12} fill={fillOf(C.s, 25)} stroke={C.s} /><Txt x={28} y={238} cls="fx-small" a="start">{T('cached K/V (reused, not recomputed)', 'K/V v cache (znovu použité, neprepočítavajú sa)')}</Txt>
      <rect x={8} y={244} width={14} height={12} fill={fillOf(C.r, 50)} stroke={C.r} /><Txt x={28} y={254} cls="fx-small" a="start">{T('new token: computed now, its K/V appended', 'nový token: počíta sa teraz, jeho K/V sa pripoja')}</Txt>
    </>),
  }),
  journey: (T) => ({
    h: 520, label: T('The journey of a chat message: app, HTTP request, server queue, chat template, tokenizer, prefill, decode loop with sampling, detokenizer, streaming back to the screen.', 'Cesta správy v chate: aplikácia, požiadavka HTTP, front na serveri, šablóna chatu, tokenizér, predvyplnenie, slučka generovania so vzorkovaním, detokenizér a streamovanie späť na obrazovku.'),
    body: (mk) => {
      const steps: [string, string, string][] = [
        [T('1 · You type a message', '1 · Napíšete správu'), T('“What is 2+2?”', '„Koľko je 2+2?“'), C.t3],
        [T('2 · App sends an HTTP request', '2 · Aplikácia pošle požiadavku HTTP'), T('JSON: messages, temperature, stream', 'JSON: správy, teplota, stream'), C.t3],
        [T('3 · Server queues and batches it', '3 · Server ju zaradí do frontu a dávky'), T('with other users’ requests', 's požiadavkami iných používateľov'), C.s],
        [T('4 · Chat template', '4 · Šablóna chatu'), '<|im_start|>user …<|im_end|>', C.act],
        [T('5 · Tokenizer', '5 · Tokenizér'), T('text → 19 token ids', 'text → 19 ID tokenov'), C.act],
        [T('6 · Prefill on the GPU', '6 · Predvyplnenie na GPU'), T('all prompt tokens, fills the K/V cache', 'všetky tokeny zadania, naplní cache K/V'), C.w],
        [T('7 · Decode loop', '7 · Slučka generovania'), T('logits → sample → next token, repeat', 'logity → vzorkovanie → ďalší token, znova'), C.a],
        [T('8 · Detokenize + stream', '8 · Detokenizácia + streamovanie'), T('each new piece is sent at once', 'každý nový kúsok sa pošle hneď'), C.act],
        [T('9 · Stop', '9 · Koniec'), T('<|im_end|> or the length limit', '<|im_end|> alebo limit dĺžky'), C.r],
      ];
      return (<>
        {steps.map(([a, b, c], i) => (
          <g key={i}>
            <Box x={30} y={8 + i * 56} w={300} h={44} c={c} label={a} sub={b} fs={12.5} />
            {i < steps.length - 1 && <Arr x1={180} y1={52 + i * 56} x2={180} y2={62 + i * 56} mk={mk} />}
          </g>
        ))}
        <path d="M330,400 C356,400 356,360 330,360" fill="none" stroke={C.a} strokeWidth={1.6} markerEnd={`url(#${mk})`} />
      </>);
    },
  }),
  'train-loop': (T) => ({
    h: 230, label: T('The training loop: take a batch of text, run the forward pass, measure the loss, compute gradients by backpropagation, update the weights, repeat.', 'Slučka trénovania: vziať dávku textu, spustiť dopredný prechod, zmerať chybu, spätným šírením vypočítať gradienty, upraviť váhy, opakovať.'),
    body: (mk) => {
      const pts: [number, number, string, string, string][] = [
        [180, 30, T('batch of text', 'dávka textu'), T('inputs + shifted targets', 'vstupy + posunuté ciele'), C.act],
        [290, 110, T('forward pass', 'dopredný prechod'), T('logits', 'logity'), C.a],
        [180, 190, T('loss', 'chyba'), T('cross-entropy', 'krížová entropia'), C.r],
        [70, 110, T('backward pass', 'spätný prechod'), T('∂loss/∂w for every w', '∂chyba/∂w pre každé w'), C.w],
      ];
      return (<>
        {pts.map(([x, y, l, s, c], i) => <Box key={i} x={x - 62} y={y - 20} w={124} h={40} c={c} label={l} sub={s} fs={12} />)}
        <Arr x1={242} y1={34} x2={282} y2={88} mk={mk} /><Arr x1={282} y1={132} x2={242} y2={182} mk={mk} />
        <Arr x1={118} y1={182} x2={78} y2={132} mk={mk} /><Arr x1={78} y1={88} x2={118} y2={36} mk={mk} />
        <Txt x={180} y={104} cls="fx-label">{T('update', 'úprava')}</Txt>
        <Txt x={180} y={120} cls="fx-sub">w ← w − η · ∂L/∂w</Txt>
        <Txt x={62} y={58} cls="fx-small">{T('optimizer', 'optimalizátor')}</Txt>
      </>);
    },
  }),
  graph: (T) => ({
    h: 236, label: T('A computational graph: the forward pass computes values left to right, the backward pass multiplies local derivatives right to left (the chain rule).', 'Výpočtový graf: dopredný prechod počíta hodnoty zľava doprava, spätný prechod násobí lokálne derivácie sprava doľava (reťazové pravidlo).'),
    body: (mk) => (<>
      <Box x={6} y={20} w={56} h={30} c={C.w} label="w = 0.4" fs={11} />
      <Box x={6} y={70} w={56} h={30} c={C.act} label="x = 1.5" fs={11} />
      <Box x={6} y={120} w={56} h={30} c={C.w} label="b = −0.1" fs={11} />
      <Arr x1={62} y1={35} x2={96} y2={78} mk={mk} /><Arr x1={62} y1={85} x2={96} y2={85} mk={mk} /><Arr x1={62} y1={135} x2={96} y2={92} mk={mk} />
      <Box x={98} y={66} w={70} h={38} c={C.a} label="z = wx+b" sub="0.5" fs={11} />
      <Arr x1={168} y1={85} x2={190} y2={85} mk={mk} />
      <Box x={192} y={66} w={66} h={38} c={C.act} label="y = SiLU" sub="0.311" fs={11} />
      <Arr x1={258} y1={85} x2={278} y2={85} mk={mk} />
      <Box x={280} y={66} w={74} h={38} c={C.r} label="(y − 0.8)²" sub="0.239" fs={11} />
      <Arr x1={340} y1={166} x2={74} y2={166} mk={mk} c={C.r} dash />
      <Txt x={210} y={160} cls="fx-small" c={C.r}>{T('backward pass (chain rule)', 'spätný prechod (reťazové pravidlo)')}</Txt>
      <Txt x={180} y={188} cls="fx-sub" c={C.r}>{T('∂L/∂y = −0.98 → ∂L/∂z = −0.72', '∂L/∂y = −0,98 → ∂L/∂z = −0,72')}</Txt>
      <Txt x={180} y={206} cls="fx-sub" c={C.r}>{T('∂L/∂w = ∂L/∂z · x = −1.09;  ∂L/∂b = −0.72', '∂L/∂w = ∂L/∂z · x = −1,09;  ∂L/∂b = −0,72')}</Txt>
      <Txt x={180} y={228} cls="fx-small">{T('negative gradient → increasing w lowers the loss', 'záporný gradient → zväčšenie w zníži chybu')}</Txt>
    </>),
  }),
  pipeline: (T) => ({
    h: 330, label: T('How Qwen3 was trained according to its report: three pre-training stages, then post-training; the small models were distilled from larger teachers.', 'Ako sa podľa správy trénoval Qwen3: tri fázy predtrénovania, potom dotrénovanie; malé modely vznikli destiláciou z väčších učiteľov.'),
    body: (mk) => {
      const st: [string, string, string][] = [
        [T('S1 general pre-training', 'S1 všeobecné predtrénovanie'), T('> 30T tokens, 4,096-token sequences', '> 30 bil. tokenov, sekvencie 4 096'), C.w],
        [T('S2 reasoning stage', 'S2 fáza uvažovania'), T('~5T tokens, more STEM and code', '~5 bil. tokenov, viac STEM a kódu'), C.w],
        [T('S3 long context', 'S3 dlhý kontext'), T('32,768 tokens, RoPE base 10k → 1M', '32 768 tokenov, základ RoPE 10k → 1M'), C.w],
        [T('Post-training (flagship models)', 'Dotrénovanie (vlajkové modely)'), T('CoT cold start → reasoning RL → mode fusion → general RL', 'CoT → RL uvažovania → zlúčenie režimov → RL'), C.a],
        [T('Strong-to-weak distillation', 'Destilácia zo silného na slabý'), T('Qwen3-8B, 30B-A3B learn from teachers’ logits', 'Qwen3-8B, 30B-A3B sa učia z logitov učiteľov'), C.act],
      ];
      return (<>
        {st.map(([a, b, c], i) => (
          <g key={i}>
            <Box x={20} y={8 + i * 64} w={320} h={48} c={c} label={a} sub={b} fs={12.5} />
            {i < st.length - 1 && <Arr x1={180} y1={56 + i * 64} x2={180} y2={70 + i * 64} mk={mk} />}
          </g>
        ))}
        <Txt x={180} y={326} cls="fx-small">{T('36T tokens in 119 languages and dialects (Qwen3 technical report)', '36 bil. tokenov v 119 jazykoch a nárečiach (technická správa Qwen3)')}</Txt>
      </>);
    },
  }),
  serving: (T) => ({
    h: 250, label: T('Serving many users: continuous batching fills free slots as soon as a request finishes, and a paged K/V cache stores each sequence in fixed-size blocks.', 'Obsluha mnohých používateľov: priebežné dávkovanie zaplní voľné miesto hneď, ako jedna požiadavka skončí, a stránkovaná cache K/V ukladá každú sekvenciu do blokov pevnej veľkosti.'),
    body: () => {
      const rows: [string, number, number, string][] = [['A', 0, 9, C.a], ['B', 0, 4, C.f], ['C', 4, 11, C.act], ['D', 9, 12, C.r]];
      return (<>
        <Txt x={8} y={16} cls="fx-label" a="start">{T('continuous batching (time →)', 'priebežné dávkovanie (čas →)')}</Txt>
        {Array.from({ length: 12 }, (_, t) => <line key={t} x1={40 + t * 26} y1={24} x2={40 + t * 26} y2={128} stroke={C.line} strokeWidth={0.5} />)}
        {rows.map(([n, a, b, c], i) => <g key={n}><text x={10} y={44 + (i % 3) * 32} className="fx-small">{T('slot', 'miesto')} {i % 3}</text><rect x={40 + a * 26} y={30 + (i % 3) * 32} width={(b - a) * 26 - 2} height={22} rx={4} fill={fillOf(c, 40)} stroke={c} /><text x={46 + a * 26} y={45 + (i % 3) * 32} className="fx-small">{T(`request ${n}`, `požiadavka ${n}`)}</text></g>)}
        <Txt x={8} y={152} cls="fx-label" a="start">{T('paged K/V cache (blocks of 16 tokens)', 'stránkovaná cache K/V (bloky po 16 tokenoch)')}</Txt>
        {Array.from({ length: 12 }, (_, i) => {
          const owner = [0, 0, 2, 1, 0, 2, 3, 2, 0, -1, 3, -1][i];
          const col = owner < 0 ? 'var(--bg-2)' : fillOf([C.a, C.f, C.act, C.r][owner], 45);
          return <g key={i}><rect x={8 + i * 29} y={162} width={26} height={30} rx={3} fill={col} stroke={owner < 0 ? C.line : [C.a, C.f, C.act, C.r][owner]} strokeDasharray={owner < 0 ? '3 3' : undefined} /><text x={21 + i * 29} y={181} textAnchor="middle" className="fx-small">{owner < 0 ? '' : 'ABCD'[owner]}</text></g>;
        })}
        <Txt x={180} y={208} cls="fx-small">{T('a sequence’s blocks need not be adjacent;', 'bloky jednej sekvencie nemusia byť vedľa seba;')}</Txt>
        <Txt x={180} y={221} cls="fx-small">{T('free blocks are reused at once', 'voľné bloky sa hneď znovu použijú')}</Txt>
        <Txt x={180} y={238} cls="fx-small">{T('(schematic; real servers such as vLLM use the same idea)', '(schéma; skutočné servery ako vLLM používajú tú istú myšlienku)')}</Txt>
      </>);
    },
  }),
  quant: (T) => ({
    h: 200, label: T('Quantization: each float weight is mapped to the nearest point of a small integer grid times a scale; the gap is the quantization error.', 'Kvantizácia: každá váha sa zobrazí na najbližší bod malej celočíselnej mriežky krát mierka; rozdiel je chyba kvantizácie.'),
    body: () => {
      const vals = [0.021, 0.036, -0.051, -0.003, 0.012, -0.027];
      const s = 0.051 / 7;
      const X = (v: number) => 180 + v * 3000;
      return (<>
        <line x1={20} y1={90} x2={340} y2={90} stroke={C.line} />
        {Array.from({ length: 15 }, (_, i) => i - 7).map((q) => <g key={q}><line x1={X(q * s)} y1={80} x2={X(q * s)} y2={100} stroke={C.t3} /><text x={X(q * s)} y={116} textAnchor="middle" className="fx-small">{q}</text></g>)}
        {vals.map((v, i) => {
          const q = Math.round(v / s);
          return <g key={i}><circle cx={X(v)} cy={60 - (i % 3) * 12} r={4} fill={C.w} /><line x1={X(v)} y1={60 - (i % 3) * 12} x2={X(q * s)} y2={88} stroke={C.r} strokeDasharray="2 2" /></g>;
        })}
        <Txt x={180} y={20} cls="fx-sub">{T('purple = original BF16 weights · ticks = 4-bit levels (−8…7)', 'fialové = pôvodné váhy BF16 · čiarky = 4-bitové úrovne (−8…7)')}</Txt>
        <Txt x={180} y={146} cls="fx-sub">{T('w ≈ q · scale, scale = max|w| / 7 per group of 64', 'w ≈ q · mierka, mierka = max|w| / 7 pre skupinu 64')}</Txt>
        <Txt x={180} y={166} cls="fx-small">{T('16 bits → 4 bits + a shared 16-bit scale: about 3.8× less memory', '16 bitov → 4 bity + spoločná 16-bitová mierka: asi 3,8× menej pamäte')}</Txt>
        <Txt x={180} y={184} cls="fx-small">{T('dashed red = rounding error', 'čiarkovaná červená = chyba zaokrúhlenia')}</Txt>
      </>);
    },
  }),
  safetensors: (T) => ({
    h: 170, label: T('Safetensors file layout: an 8-byte header length, a JSON header with name, dtype, shape and byte offsets of every tensor, then all tensor bytes.', 'Rozloženie súboru safetensors: 8-bajtová dĺžka hlavičky, hlavička JSON s názvom, typom, tvarom a bajtovými posunmi každého tenzora a potom všetky bajty tenzorov.'),
    body: (mk) => (<>
      <Box x={8} y={20} w={52} h={44} c={C.r} label="8 B" sub={T('length n', 'dĺžka n')} fs={12} />
      <Box x={62} y={20} w={110} h={44} c={C.act} label={T('JSON header', 'hlavička JSON')} sub={T('n bytes', 'n bajtov')} fs={12} />
      <Box x={174} y={20} w={178} h={44} c={C.w} label={T('tensor bytes (BF16)', 'bajty tenzorov (BF16)')} sub={T('row-major, 2 B per number', 'po riadkoch, 2 B na číslo')} fs={12} />
      <Arr x1={117} y1={66} x2={117} y2={86} mk={mk} />
      <rect x={8} y={88} width={344} height={66} rx={6} fill={C.bg} stroke={C.line} />
      <text x={16} y={106} className="fx-small fx-mono">{'"model.layers.0.self_attn.q_proj.weight": {'}</text>
      <text x={24} y={122} className="fx-small fx-mono">{'"dtype": "BF16", "shape": [2048, 1024],'}</text>
      <text x={24} y={138} className="fx-small fx-mono">{'"data_offsets": [start, end] }'}</text>
      <Txt x={180} y={166} cls="fx-small">{T('element [r, c] lives at 8 + n + start + 2·(r·cols + c)', 'prvok [r, c] je na 8 + n + start + 2·(r·stĺpce + c)')}</Txt>
    </>),
  }),
  overview: (T) => ({
    h: 300, label: T('The whole transformer on one page: tokens, embedding, repeated blocks of attention and feed-forward with residual adds, final norm, output head, sampling.', 'Celý transformer na jednej strane: tokeny, embedding, opakované bloky pozornosti a doprednej siete s reziduálnymi sčítaniami, záverečná normalizácia, výstupná hlava, vzorkovanie.'),
    body: (mk) => {
      const L: [string, string, string][] = [
        [T('text → tokens', 'text → tokeny'), T('tokenizer, chat template', 'tokenizér, šablóna chatu'), C.act],
        [T('tokens → vectors', 'tokeny → vektory'), T('embedding lookup', 'vyhľadanie embeddingu'), C.w],
        [T('× N blocks: mix tokens', '× N blokov: miešanie tokenov'), T('RMSNorm → attention (Q·K, softmax, ·V) → +', 'RMSNorm → pozornosť (Q·K, softmax, ·V) → +'), C.a],
        [T('× N blocks: process each token', '× N blokov: spracovanie tokenu'), T('RMSNorm → SwiGLU or MoE → +', 'RMSNorm → SwiGLU alebo MoE → +'), C.f],
        [T('vector → scores', 'vektor → skóre'), T('final RMSNorm, lm_head, softmax', 'záverečná RMSNorm, lm_head, softmax'), C.n],
        [T('scores → next token', 'skóre → ďalší token'), T('temperature, top-k, top-p, draw', 'teplota, top-k, top-p, losovanie'), C.r],
      ];
      return (<>{L.map(([a, b, c], i) => (
        <g key={i}><Box x={20} y={6 + i * 49} w={320} h={40} c={c} label={a} sub={b} fs={12.5} />{i < L.length - 1 && <Arr x1={180} y1={46 + i * 49} x2={180} y2={54 + i * 49} mk={mk} />}</g>
      ))}</>);
    },
  }),
};

export const FIGURE_IDS = Object.keys(FIGS) as FigureId[];

export function Figure({ id, caption }: { id: FigureId; caption?: string }) {
  const { lang } = useLang();
  const T: TT = (en, sk) => (lang === 'sk' ? sk : en);
  const f = FIGS[id](T);
  return (
    <figure className="fig" data-figure={id}>
      <Svg h={f.h} label={f.label}>{f.body}</Svg>
      <figcaption>🖼️ {caption ?? f.label}</figcaption>
    </figure>
  );
}
