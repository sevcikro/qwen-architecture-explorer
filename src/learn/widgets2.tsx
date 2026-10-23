import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import { useLang } from '../i18n/i18n';
import { Frame, HBars, Range, pick } from './kit';
import { useJourney, useSamples } from '../data/data';
import { QWEN3_8B } from '../models/profiles';
import { toBf16Bits, fromBf16Bits } from '../math/bf16';

// Course experiments added in v1.2 (vectors, cross-entropy, RoPE, GQA, K/V cache cost, the message journey,
// backpropagation, quantization). Every label exists in English and Slovak.

export type WidgetId2 = 'dot2d' | 'xent' | 'rope2d' | 'gqa' | 'kvcost' | 'journey' | 'backprop' | 'quant' | 'context';

const deg = (d: number) => (d * Math.PI) / 180;

// ---------------------------------------------------------------------------------------------
function Dot2d() {
  const { lang, fnum } = useLang();
  const [a1, setA1] = useState(30), [a2, setA2] = useState(75), [l1, setL1] = useState(1.6), [l2, setL2] = useState(1.2);
  const A = [l1 * Math.cos(deg(a1)), l1 * Math.sin(deg(a1))], B = [l2 * Math.cos(deg(a2)), l2 * Math.sin(deg(a2))];
  const dot = A[0] * B[0] + A[1] * B[1];
  const cos = dot / (l1 * l2);
  const S = 60, cx = 110, cy = 110;
  const P = (v: number[]) => [cx + v[0] * S, cy - v[1] * S];
  const [ax, ay] = P(A), [bx, by] = P(B);
  return (
    <Frame title={pick(lang, 'dot product and angle', 'skalárny súčin a uhol')} ev={[]} note={pick(lang, 'Attention scores, similarity between embeddings and every neuron are built from dot products. Same direction → large positive; perpendicular → 0; opposite → negative.', 'Skóre pozornosti, podobnosť embeddingov aj každý neurón stoja na skalárnych súčinoch. Rovnaký smer → veľké kladné; kolmé → 0; opačné → záporné.')}>
      <div className="grid2">
        <svg viewBox="0 0 220 220" style={{ maxWidth: 260, width: '100%' }} role="img" aria-label={pick(lang, `two vectors with angle ${Math.abs(a2 - a1)} degrees, dot product ${dot.toFixed(2)}`, `dva vektory s uhlom ${Math.abs(a2 - a1)} stupňov, skalárny súčin ${dot.toFixed(2)}`)}>
          <circle cx={cx} cy={cy} r={S} fill="none" stroke="var(--line)" strokeDasharray="3 3" />
          <line x1={10} y1={cy} x2={210} y2={cy} stroke="var(--line)" /><line x1={cx} y1={10} x2={cx} y2={210} stroke="var(--line)" />
          <line x1={cx} y1={cy} x2={ax} y2={ay} stroke="var(--attn)" strokeWidth={3} /><circle cx={ax} cy={ay} r={4} fill="var(--attn)" />
          <line x1={cx} y1={cy} x2={bx} y2={by} stroke="var(--router)" strokeWidth={3} /><circle cx={bx} cy={by} r={4} fill="var(--router)" />
          <text x={ax + 6} y={ay} fontSize={13} fill="var(--attn)">a</text><text x={bx + 6} y={by} fontSize={13} fill="var(--router)">b</text>
        </svg>
        <div>
          <Range label={pick(lang, 'angle of a (°)', 'uhol a (°)')} value={a1} min={0} max={360} step={1} onChange={setA1} fmt={(v) => String(v)} />
          <Range label={pick(lang, 'angle of b (°)', 'uhol b (°)')} value={a2} min={0} max={360} step={1} onChange={setA2} fmt={(v) => String(v)} />
          <Range label={pick(lang, 'length of a', 'dĺžka a')} value={l1} min={0.2} max={1.8} step={0.1} onChange={setL1} fmt={(v) => fnum(v, 1)} />
          <Range label={pick(lang, 'length of b', 'dĺžka b')} value={l2} min={0.2} max={1.8} step={0.1} onChange={setL2} fmt={(v) => fnum(v, 1)} />
          <p className="mono" style={{ fontSize: '0.9rem' }}>a = ({fnum(A[0], 2)}, {fnum(A[1], 2)}) · b = ({fnum(B[0], 2)}, {fnum(B[1], 2)})<br />a·b = a₀b₀ + a₁b₁ = <strong>{fnum(dot, 3)}</strong><br />cos θ = {fnum(cos, 3)}</p>
        </div>
      </div>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function Xent() {
  const { lang, fnum } = useLang();
  const words = ['Paris', 'London', 'banana'];
  const [z, setZ] = useState([3, 1.5, -1]);
  const [target, setTarget] = useState(0);
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / s);
  const loss = -Math.log(p[target]);
  const W = 300, H = 120;
  const curve = Array.from({ length: 60 }, (_, i) => { const q = 0.01 + (i / 59) * 0.99; return `${(q * W).toFixed(1)},${(H - Math.min(H, (-Math.log(q) / 4.7) * H)).toFixed(1)}`; }).join(' ');
  return (
    <Frame title={pick(lang, 'softmax and the training loss', 'softmax a chyba pri trénovaní')} ev={[]} note={pick(lang, 'Training pushes the logit of the correct next token up and the others down, because that lowers −log p. A confident wrong guess is punished hard.', 'Trénovanie zvyšuje logit správneho ďalšieho tokenu a ostatné znižuje, lebo to zmenšuje −log p. Sebaistý chybný odhad je trestaný veľmi tvrdo.')}>
      <div className="row">
        {words.map((w, i) => <Range key={w} label={`logit “${w}”`} value={z[i]} min={-4} max={6} step={0.1} onChange={(v) => setZ(z.map((o, k) => (k === i ? v : o)))} fmt={(v) => fnum(v, 1)} />)}
      </div>
      <div className="row" role="radiogroup" aria-label={pick(lang, 'correct next token', 'správny ďalší token')} style={{ margin: '6px 0' }}>
        <span className="faint">{pick(lang, 'correct answer:', 'správna odpoveď:')}</span>
        {words.map((w, i) => <button key={w} className="small" role="radio" aria-checked={target === i} onClick={() => setTarget(i)}>{w}</button>)}
      </div>
      <HBars max={1} items={words.map((w, i) => ({ label: w, v: p[i], on: i === target, extra: fnum(p[i], 3) }))} />
      <svg viewBox={`0 0 ${W} ${H + 18}`} style={{ width: '100%', maxWidth: 420, marginTop: 8 }} role="img" aria-label={pick(lang, `loss curve −log p; current loss ${loss.toFixed(3)}`, `krivka chyby −log p; aktuálna chyba ${loss.toFixed(3)}`)}>
        <polyline points={curve} fill="none" stroke="var(--router)" strokeWidth={2} />
        <circle cx={p[target] * W} cy={H - Math.min(H, (loss / 4.7) * H)} r={5} fill="var(--attn)" />
        <text x={4} y={H + 14} fontSize={10} fill="var(--text-3)">p = 0</text><text x={W - 4} y={H + 14} fontSize={10} fill="var(--text-3)" textAnchor="end">p = 1</text>
        <text x={W / 2} y={14} fontSize={11} fill="var(--text-2)" textAnchor="middle">{pick(lang, 'loss = −log p(correct)', 'chyba = −log p(správne)')}</text>
      </svg>
      <p className="mono">p({words[target]}) = {fnum(p[target], 4)} → {pick(lang, 'loss', 'chyba')} = −ln p = <strong>{fnum(loss, 4)}</strong></p>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function Rope2d() {
  const { lang, fnum } = useLang();
  const [m, setM] = useState(5), [n, setN] = useState(2), [fast, setFast] = useState(true);
  const theta = fast ? 0.5 : 0.08;
  const q = [1, 0.3], k = [0.6, 0.8];
  const rot = (v: number[], a: number) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
  const qm = rot(q, m * theta), kn = rot(k, n * theta);
  const score = qm[0] * kn[0] + qm[1] * kn[1];
  const shifted = (() => { const a = rot(q, (m + 10) * theta), b = rot(k, (n + 10) * theta); return a[0] * b[0] + a[1] * b[1]; })();
  const S = 70, c = 100;
  return (
    <Frame title={pick(lang, 'RoPE: rotate by position', 'RoPE: otočenie podľa pozície')} ev={[]} note={pick(lang, 'One pair of coordinates, as RoPE treats it. Qwen3 rotates 64 such pairs per head, each with its own speed θᵢ; RoPE has no learned parameters.', 'Jeden pár súradníc, ako s ním pracuje RoPE. Qwen3 otáča 64 takých párov v každej hlave, každý vlastnou rýchlosťou θᵢ; RoPE nemá naučené parametre.')}>
      <div className="grid2">
        <svg viewBox="0 0 200 200" style={{ maxWidth: 240, width: '100%' }} role="img" aria-label={pick(lang, `query at position ${m}, key at position ${n}, score ${score.toFixed(3)}`, `dopyt na pozícii ${m}, kľúč na pozícii ${n}, skóre ${score.toFixed(3)}`)}>
          <circle cx={c} cy={c} r={S} fill="none" stroke="var(--line)" strokeDasharray="3 3" />
          <line x1={c} y1={c} x2={c + qm[0] * S} y2={c - qm[1] * S} stroke="var(--attn)" strokeWidth={3} />
          <line x1={c} y1={c} x2={c + kn[0] * S} y2={c - kn[1] * S} stroke="var(--router)" strokeWidth={3} />
          <text x={c + qm[0] * S + 4} y={c - qm[1] * S} fontSize={12} fill="var(--attn)">q@{m}</text>
          <text x={c + kn[0] * S + 4} y={c - kn[1] * S} fontSize={12} fill="var(--router)">k@{n}</text>
        </svg>
        <div>
          <Range label={pick(lang, 'query position m', 'pozícia dopytu m')} value={m} min={0} max={40} step={1} onChange={setM} fmt={(v) => String(v)} />
          <Range label={pick(lang, 'key position n', 'pozícia kľúča n')} value={n} min={0} max={40} step={1} onChange={setN} fmt={(v) => String(v)} />
          <div className="row" role="group" aria-label={pick(lang, 'rotation speed', 'rýchlosť otáčania')}>
            <button className="small" aria-pressed={fast} onClick={() => setFast(true)}>{pick(lang, 'fast pair', 'rýchly pár')}</button>
            <button className="small" aria-pressed={!fast} onClick={() => setFast(false)}>{pick(lang, 'slow pair', 'pomalý pár')}</button>
          </div>
          <p className="mono" style={{ fontSize: '0.9rem' }}>{pick(lang, 'distance', 'vzdialenosť')} m − n = {m - n}<br />{pick(lang, 'score', 'skóre')} q_m · k_n = <strong>{fnum(score, 4)}</strong><br />{pick(lang, 'both moved +10', 'obe posunuté o +10')}: {fnum(shifted, 4)} {Math.abs(shifted - score) < 1e-9 ? '✓' : ''}</p>
        </div>
      </div>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function Gqa() {
  const { lang, num } = useLang();
  const opts = [1, 2, 4, 8, 16, 32];
  const [hkv, setHkv] = useState(8);
  const p = QWEN3_8B;
  const perTok = 2 * p.layers * hkv * p.headDim * 2;
  const ctx = 32768;
  return (
    <Frame title={pick(lang, 'how many K/V heads?', 'koľko hláv K/V?')} ev={['derived']} note={pick(lang, `Qwen3-8B dimensions (36 blocks, 32 query heads, head width 128, BF16). Qwen3-8B actually uses 8 K/V heads. 32 = ordinary multi-head attention; 1 = multi-query attention.`, `Rozmery Qwen3-8B (36 blokov, 32 hláv dopytov, šírka hlavy 128, BF16). Qwen3-8B v skutočnosti používa 8 hláv K/V. 32 = bežná viachlavová pozornosť; 1 = pozornosť s jedinou hlavou K/V pre všetky dopyty (MQA).`)}>
      <div className="row" role="radiogroup" aria-label={pick(lang, 'number of K/V heads', 'počet hláv K/V')}>
        {opts.map((o) => <button key={o} className="small" role="radio" aria-checked={hkv === o} onClick={() => setHkv(o)}>{o}{o === 8 ? ' ★' : ''}</button>)}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(32, 1fr)', gap: 2, margin: '10px 0' }} aria-hidden="true">
        {Array.from({ length: 32 }, (_, h) => { const g = Math.floor(h / (32 / hkv)); return <span key={h} style={{ height: 22, borderRadius: 3, background: `hsl(${(g * 360) / hkv} 55% 55%)` }} title={`q${h} → kv${g}`} />; })}
      </div>
      <p className="mono" style={{ fontSize: '0.9rem' }}>{pick(lang, 'query heads per K/V head', 'hláv dopytov na hlavu K/V')}: {32 / hkv}<br />{pick(lang, 'K/V cache per token', 'cache K/V na token')}: 2 × 36 × {hkv} × 128 × 2 B = <strong>{num(perTok)} B</strong><br />{pick(lang, `at ${num(ctx)} tokens`, `pri ${num(ctx)} tokenoch`)}: <strong>{num((perTok * ctx) / 2 ** 30, 2)} GiB</strong></p>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function KvCost() {
  const { lang, num } = useLang();
  const [P, setP] = useState(500), [G, setG] = useState(500);
  // The first answer token comes from the prompt pass; each further token needs one more pass.
  // Without a cache, the pass that produces token i re-runs all P + i tokens.
  let noCache = 0;
  for (let i = 0; i < G; i++) noCache += P + i;
  const withCache = P + (G - 1);
  const max = Math.max(noCache, withCache);
  return (
    <Frame title={pick(lang, 'why the K/V cache matters', 'prečo je cache K/V dôležitá')} ev={['derived']} note={pick(lang, 'Counted in “token passes” (one token through all blocks). Without a cache, every new token re-runs the whole prefix; with a cache, each token passes once and attention reads the stored K/V instead. Attention reading still grows with length in both cases.', 'Počítané v „prechodoch tokenu“ (jeden token všetkými blokmi). Bez cache každý nový token znovu prepočíta celú predponu; s cache prejde každý token raz a pozornosť namiesto toho číta uložené K/V. Čítanie v pozornosti rastie s dĺžkou v oboch prípadoch.')}>
      <div className="row">
        <Range label={pick(lang, 'prompt tokens', 'tokeny zadania')} value={P} min={10} max={4000} step={10} onChange={setP} fmt={(v) => num(v)} />
        <Range label={pick(lang, 'generated tokens', 'generované tokeny')} value={G} min={1} max={4000} step={1} onChange={setG} fmt={(v) => num(v)} />
      </div>
      <HBars max={max} items={[
        { label: pick(lang, 'no cache', 'bez cache'), v: noCache, extra: num(noCache) },
        { label: pick(lang, 'with cache', 's cache'), v: withCache, extra: num(withCache) },
      ]} />
      <p className="mono" style={{ fontSize: '0.9rem' }}>{pick(lang, 'savings', 'úspora')}: {num(noCache / withCache, 1)}×</p>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
const STAGES = 9;
function Journey() {
  const { lang, fnum, num } = useLang();
  const T = (en: string, sk: string) => pick(lang, en, sk);
  const j = useJourney();
  const [stage, setStage] = useState(0);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const nSteps = j.status === 'ok' ? j.data.steps.length : 0;
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setTimeout(() => {
      if (stage < 6) setStage(stage + 1);
      else if (stage === 6 && k < nSteps - 1) setK(k + 1);
      else if (stage < STAGES - 1) setStage(stage + 1);
      else setPlaying(false);
    }, stage === 6 ? (reduce ? 400 : 120) : 1300);
    return () => window.clearTimeout(timer.current);
  }, [playing, stage, k, nSteps, reduce]);
  if (j.status !== 'ok') return <Frame title={T('the journey of one message', 'cesta jednej správy')} ev={['sample']}><p className="muted">{j.status === 'loading' ? T('Loading the recorded run…', 'Načítavam zaznamenaný beh…') : T('The recorded run could not be loaded.', 'Zaznamenaný beh sa nepodarilo načítať.')}</p></Frame>;
  const d = j.data;
  const titles = [
    T('1 · You type a message', '1 · Napíšete správu'), T('2 · The app sends an HTTP request', '2 · Aplikácia pošle požiadavku HTTP'), T('3 · The server schedules it', '3 · Server ju naplánuje'),
    T('4 · Chat template', '4 · Šablóna chatu'), T('5 · Tokenizer', '5 · Tokenizér'), T('6 · Prefill', '6 · Predvyplnenie'),
    T('7 · Decode loop', '7 · Slučka generovania'), T('8 · Detokenize and stream', '8 · Detokenizácia a streamovanie'), T('9 · Stop', '9 · Koniec'),
  ];
  const step = d.steps[Math.min(k, nSteps - 1)];
  const textSoFar = d.steps.slice(0, stage >= 7 ? nSteps : k + 1).map((s) => s.piece).join('').replace('<|im_end|>', '');
  const body: Record<number, ReactElement> = {
    0: <p>{T('You type into a chat app:', 'Do chatovej aplikácie napíšete:')} <strong>“{d.messages[0].content}”</strong>. {T('Nothing about the model has changed yet; this is just text on your device.', 'Na modeli sa zatiaľ nič nezmenilo; je to len text vo vašom zariadení.')}</p>,
    1: <><p>{T('The app sends the whole conversation (not only the last message) to a server, typically as JSON over HTTPS:', 'Aplikácia pošle na server celý rozhovor (nielen poslednú správu), zvyčajne ako JSON cez HTTPS:')}</p><pre className="code">{JSON.stringify({ model: 'Qwen/Qwen3-0.6B', messages: d.messages, stream: true, max_tokens: 512 }, null, 2)}</pre></>,
    2: <p>{T('The server puts the request in a queue. A GPU is most efficient when it works on many conversations at once, so requests from many users are batched together and new ones join as soon as a slot frees up (continuous batching). The model weights are loaded once and shared by all of them; each conversation gets its own K/V cache.', 'Server zaradí požiadavku do frontu. GPU je najefektívnejšia, keď pracuje na mnohých rozhovoroch naraz, preto sa požiadavky mnohých používateľov spájajú do dávok a nové sa pridávajú hneď, ako sa uvoľní miesto (priebežné dávkovanie). Váhy modelu sú načítané raz a zdieľajú ich všetky; každý rozhovor má vlastnú cache K/V.')}</p>,
    3: <><p>{T('The messages are wrapped in the model’s chat template, which marks who speaks with special tokens. With thinking disabled, Qwen3 also inserts an empty think block:', 'Správy sa zabalia do šablóny chatu modelu, ktorá špeciálnymi tokenmi označí, kto hovorí. Pri vypnutom premýšľaní Qwen3 vloží aj prázdny blok premýšľania:')}</p><pre className="code">{d.renderedPrompt}</pre></>,
    4: <><p>{T(`The tokenizer turns that text into ${d.promptIds.length} token ids (special tokens in orange):`, `Tokenizér premení tento text na ${d.promptIds.length} ID tokenov (špeciálne tokeny oranžovo):`)}</p><div className="row" style={{ gap: 4 }}>{d.promptPieces.map((pc, i) => <span key={i} className={`chip${d.promptIds[i] >= 151643 ? ' special' : ''}`}><span>{JSON.stringify(pc).slice(1, -1) || '∅'}</span><span className="faint">{d.promptIds[i]}</span></span>)}</div></>,
    5: <p>{T(`Prefill: all ${d.promptIds.length} prompt tokens go through all 28 blocks of Qwen3-0.6B in one parallel pass. Every block stores the keys and values of every prompt position in the K/V cache. The last position’s logits give the first answer token. Prefill decides the “time to first token”.`, `Predvyplnenie: všetkých ${d.promptIds.length} tokenov zadania prejde všetkými 28 blokmi Qwen3-0.6B v jednom paralelnom prechode. Každý blok uloží kľúče a hodnoty každej pozície zadania do cache K/V. Logity poslednej pozície dajú prvý token odpovede. Predvyplnenie určuje „čas do prvého tokenu“.`)}</p>,
    6: <>
      <p>{k === 0
        ? T(`Answer token 1 of ${nSteps}: it comes straight from the prefill pass (the logits of the last prompt position). The model outputs a probability for all 151,936 rows; this run picks the most probable one (greedy).`, `Token odpovede 1 z ${nSteps}: pochádza priamo z predvyplnenia (logity poslednej pozície zadania). Model vypíše pravdepodobnosť pre všetkých 151 936 riadkov; tento beh vyberie najpravdepodobnejší (hladovo).`)
        : T(`Answer token ${k + 1} of ${nSteps}: one decode pass of only the previous token “${JSON.stringify(d.steps[k - 1].piece).slice(1, -1)}”. In every block its query attends to ${d.promptIds.length + k} positions: ${d.promptIds.length + k - 1} keys and values read from the cache plus its own, which are then appended. The model outputs a probability for all 151,936 rows; this run picks the most probable one (greedy).`, `Token odpovede ${k + 1} z ${nSteps}: jeden prechod generovania len s predchádzajúcim tokenom „${JSON.stringify(d.steps[k - 1].piece).slice(1, -1)}“. V každom bloku jeho dopyt venuje pozornosť ${d.promptIds.length + k} pozíciám: ${d.promptIds.length + k - 1} kľúčom a hodnotám z cache plus vlastným, ktoré sa potom pripoja. Model vypíše pravdepodobnosť pre všetkých 151 936 riadkov; tento beh vyberie najpravdepodobnejší (hladovo).`)}</p>
      <HBars max={1} items={step.top.map((c, i) => ({ label: JSON.stringify(c.piece).slice(1, -1), v: c.p, on: i === 0, extra: fnum(c.p, 3) }))} />
      <Range label={T('answer token', 'token odpovede')} value={k} min={0} max={nSteps - 1} step={1} onChange={(v) => { setK(v); setPlaying(false); }} fmt={(v) => `${v + 1}`} />
    </>,
    7: <p>{T('Each chosen token id is turned back into text (detokenized) and sent to your screen immediately, which is why answers appear word by word. Some tokens are only part of a word or a single byte of a multi-byte character, so servers buffer until the text is valid.', 'Každé vybrané ID tokenu sa premení späť na text (detokenizuje) a hneď sa pošle na obrazovku, preto sa odpovede objavujú slovo po slove. Niektoré tokeny sú len časťou slova alebo jedným bajtom viacbajtového znaku, preto servery čakajú, kým je text platný.')}</p>,
    8: <p>{d.stoppedOnEnd ? T(`The model produced <|im_end|> after ${nSteps} tokens, which ends its turn. The K/V cache of this conversation can now be freed (or kept for the next message). The weights never changed.`, `Model po ${nSteps} tokenoch vytvoril <|im_end|>, čím ukončil svoj ťah. Cache K/V tohto rozhovoru sa teraz môže uvoľniť (alebo ponechať pre ďalšiu správu). Váhy sa nikdy nezmenili.`) : T('The run stopped at its length limit.', 'Beh sa zastavil na limite dĺžky.')}</p>,
  };
  return (
    <Frame title={T('the journey of one message (real Qwen3-0.6B run)', 'cesta jednej správy (skutočný beh Qwen3-0.6B)')} ev={['sample']} note={T(`Recorded with ${d.library}, ${d.dtype}, ${d.decoding}, ${d.model} @ ${d.revision.slice(0, 12)}, on ${d.recorded}. Server-side steps 2–3 and 8 are explained, not recorded.`, `Zaznamenané s ${d.library}, ${d.dtype}, hladové dekódovanie s cache K/V, ${d.model} @ ${d.revision.slice(0, 12)}, dňa ${d.recorded}. Kroky na serveri 2–3 a 8 sú vysvetlené, nie zaznamenané.`)}>
      <div className="timeline" aria-label={T('Stages', 'Fázy')}>
        {titles.map((t, i) => <button key={i} className={`step${i === stage ? ' cur' : i < stage ? ' done' : ''}`} aria-current={i === stage ? 'step' : undefined} onClick={() => { setStage(i); setPlaying(false); }}>{t}</button>)}
      </div>
      <div className="row" style={{ margin: '8px 0' }}>
        <button onClick={() => { setStage(Math.max(0, stage - 1)); setPlaying(false); }} disabled={stage === 0}>◀ {T('Back', 'Späť')}</button>
        <button className="primary" aria-pressed={playing} onClick={() => { if (stage === STAGES - 1) { setStage(0); setK(0); } setPlaying(!playing); }}>{playing ? `❚❚ ${T('Pause', 'Pozastaviť')}` : `▶ ${T('Play', 'Prehrať')}`}</button>
        <button onClick={() => { setStage(Math.min(STAGES - 1, stage + 1)); setPlaying(false); }} disabled={stage === STAGES - 1}>{T('Next', 'Ďalej')} ▶</button>
      </div>
      <h4 aria-live="polite" style={{ margin: '6px 0' }}>{titles[stage]}</h4>
      {body[stage]}
      {stage >= 6 && (
        <div className="card tight" style={{ marginTop: 8 }}>
          <div className="faint" style={{ fontSize: '0.8rem' }}>{T('Answer on your screen so far', 'Odpoveď na obrazovke zatiaľ')} ({num(stage >= 7 ? nSteps : k + 1)} {T('tokens', 'tokenov')}):</div>
          <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>{textSoFar}<span className="cursor" aria-hidden="true">▍</span></div>
        </div>
      )}
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function Backprop() {
  const { lang, fnum } = useLang();
  const T = (en: string, sk: string) => pick(lang, en, sk);
  const [w, setW] = useState(0.4), [b, setB] = useState(-0.1), [lr, setLr] = useState(0.1);
  const [hist, setHist] = useState<number[]>([]);
  const x = 1.5, t = 0.8;
  const sig = (v: number) => 1 / (1 + Math.exp(-v));
  const f = (wv: number, bv: number) => { const z = wv * x + bv; const y = z * sig(z); return { z, y, L: (y - t) ** 2 }; };
  const { z, y, L } = f(w, b);
  const dLdy = 2 * (y - t), dydz = sig(z) * (1 + z * (1 - sig(z))), dLdz = dLdy * dydz, dLdw = dLdz * x, dLdb = dLdz;
  const h = 1e-6, num = (f(w + h, b).L - f(w - h, b).L) / (2 * h);
  return (
    <Frame title={T('backpropagation on a tiny graph', 'spätné šírenie na malom grafe')} ev={[]} note={T('z = w·x + b, y = SiLU(z), loss = (y − target)². x = 1.5, target = 0.8. The same chain rule runs through billions of weights in a real model.', 'z = w·x + b, y = SiLU(z), chyba = (y − cieľ)². x = 1,5, cieľ = 0,8. To isté reťazové pravidlo prebieha v skutočnom modeli cez miliardy váh.')}>
      <div className="row">
        <Range label="w" value={w} min={-2} max={2} step={0.01} onChange={setW} fmt={(v) => fnum(v, 2)} />
        <Range label="b" value={b} min={-2} max={2} step={0.01} onChange={setB} fmt={(v) => fnum(v, 2)} />
        <Range label={T('learning rate', 'rýchlosť učenia')} value={lr} min={0.01} max={1} step={0.01} onChange={setLr} fmt={(v) => fnum(v, 2)} />
      </div>
      <div className="table-wrap" tabIndex={0}><table className="t">
        <thead><tr><th>{T('node', 'uzol')}</th><th className="num">{T('forward value', 'hodnota dopredu')}</th><th className="num">{T('gradient ∂loss/∂node', 'gradient ∂chyba/∂uzol')}</th></tr></thead>
        <tbody>
          <tr><td>loss</td><td className="num">{fnum(L, 5)}</td><td className="num">1</td></tr>
          <tr><td>y = SiLU(z)</td><td className="num">{fnum(y, 5)}</td><td className="num">{fnum(dLdy, 5)}</td></tr>
          <tr><td>z = w·x + b</td><td className="num">{fnum(z, 5)}</td><td className="num">{fnum(dLdz, 5)}</td></tr>
          <tr><td>w</td><td className="num">{fnum(w, 3)}</td><td className="num"><strong>{fnum(dLdw, 5)}</strong></td></tr>
          <tr><td>b</td><td className="num">{fnum(b, 3)}</td><td className="num"><strong>{fnum(dLdb, 5)}</strong></td></tr>
        </tbody>
      </table></div>
      <p className="mono" style={{ fontSize: '0.86rem' }}>{T('finite-difference check', 'kontrola konečnou diferenciou')} ∂L/∂w ≈ {fnum(num, 5)} {Math.abs(num - dLdw) < 1e-5 ? '✓' : '✗'}</p>
      <div className="row">
        <button className="primary" onClick={() => { setHist([L, ...hist].slice(0, 8)); setW(w - lr * dLdw); setB(b - lr * dLdb); }}>{T('Take one gradient step', 'Urobiť jeden krok gradientu')}</button>
        <button onClick={() => { setW(0.4); setB(-0.1); setHist([]); }}>{T('Reset', 'Obnoviť')}</button>
      </div>
      {hist.length > 0 && <p className="mono" style={{ fontSize: '0.86rem' }}>{T('loss history', 'história chyby')}: {[L, ...hist].reverse().map((v) => fnum(v, 4)).join(' → ')}</p>}
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function Quant() {
  const { lang, fnum } = useLang();
  const T = (en: string, sk: string) => pick(lang, en, sk);
  const s = useSamples();
  const [bits, setBits] = useState(4);
  const tile = s.status === 'ok' ? s.data.samples.find((x) => x.modelKey === 'qwen3-8b' && x.tensor === 'model.layers.0.mlp.down_proj.weight') : undefined;
  const row = useMemo(() => tile?.values[0].slice(0, 16) ?? [], [tile]);
  if (!tile) return <Frame title={T('quantizing real weights', 'kvantizácia skutočných váh')} ev={['sample']}><p className="muted">{s.status === 'error' ? T('The real sample could not be loaded.', 'Skutočnú vzorku sa nepodarilo načítať.') : T('Loading the real sample…', 'Načítavam skutočnú vzorku…')}</p></Frame>;
  const qmax = 2 ** (bits - 1) - 1;
  const scale = fromBf16Bits(toBf16Bits(Math.max(...row.map(Math.abs)) / qmax));
  const q = row.map((v) => Math.max(-qmax - 1, Math.min(qmax, Math.round(v / scale))));
  const back = q.map((v) => v * scale);
  const err = row.map((v, i) => Math.abs(v - back[i]));
  const maxAbs = Math.max(...row.map(Math.abs));
  const bitsTotal = row.length * bits + 16;
  return (
    <Frame title={T('quantizing 16 real weights of Qwen3-8B', 'kvantizácia 16 skutočných váh Qwen3-8B')} ev={['sample']} note={T(`Values: model.layers.0.mlp.down_proj.weight[0, 0:16] read from the pinned checkpoint. One shared scale for these 16 numbers (stored in 16 bits). Real formats (GPTQ, AWQ, GGUF k-quants) use smarter rounding and grouping.`, `Hodnoty: model.layers.0.mlp.down_proj.weight[0, 0:16] prečítané z pripnutého checkpointu. Jedna spoločná mierka pre týchto 16 čísel (uložená v 16 bitoch). Skutočné formáty (GPTQ, AWQ, GGUF k-quants) používajú múdrejšie zaokrúhľovanie a zoskupovanie.`)}>
      <Range label={T('bits per weight', 'bity na váhu')} value={bits} min={2} max={8} step={1} onChange={setBits} fmt={(v) => String(v)} />
      <svg viewBox="0 0 320 140" style={{ width: '100%', maxWidth: 520 }} role="img" aria-label={T(`original and ${bits}-bit values of 16 weights`, `pôvodné a ${bits}-bitové hodnoty 16 váh`)}>
        <line x1={0} y1={70} x2={320} y2={70} stroke="var(--line)" />
        {row.map((v, i) => (
          <g key={i}>
            <rect x={4 + i * 20} y={v >= 0 ? 70 - (v / maxAbs) * 60 : 70} width={8} height={(Math.abs(v) / maxAbs) * 60} fill="var(--weight)" />
            <rect x={12 + i * 20} y={back[i] >= 0 ? 70 - (back[i] / maxAbs) * 60 : 70} width={8} height={(Math.abs(back[i]) / maxAbs) * 60} fill="var(--router)" />
          </g>
        ))}
      </svg>
      <p className="faint" style={{ fontSize: '0.82rem' }}><span style={{ color: 'var(--weight)' }}>■</span> {T('original BF16', 'pôvodné BF16')} · <span style={{ color: 'var(--router)' }}>■</span> {T(`${bits}-bit, dequantized`, `${bits}-bitové, späť prevedené`)}</p>
      <p className="mono" style={{ fontSize: '0.86rem' }}>{T('integers', 'celé čísla')}: [{q.join(', ')}]<br />scale = {fnum(scale, 3)} · {T('max error', 'max. chyba')} = {fnum(Math.max(...err), 3)} · {T('mean error', 'priem. chyba')} = {fnum(err.reduce((a, b) => a + b, 0) / err.length, 3)}<br />{T('storage', 'uloženie')}: {bitsTotal} {T('bits instead of', 'bitov namiesto')} {row.length * 16} ({fnum((row.length * 16) / bitsTotal, 2)}× {T('smaller', 'menej')})</p>
    </Frame>
  );
}

// ---------------------------------------------------------------------------------------------
function ContextBudget() {
  const { lang, num, fnum } = useLang();
  const T = (en: string, sk: string) => pick(lang, en, sk);
  const [prompt, setPrompt] = useState(6000), [out, setOut] = useState(4000), [yarn, setYarn] = useState(false);
  const win = yarn ? 131072 : 32768;
  const total = prompt + out;
  const over = total > win;
  const kv = 2 * QWEN3_8B.layers * QWEN3_8B.hkv * QWEN3_8B.headDim * 2 * Math.min(total, win);
  const W = 320;
  const px = (n: number) => Math.min(W, (n / win) * W);
  return (
    <Frame title={T('fit a conversation into the window', 'vtesnajte rozhovor do okna')} ev={['derived']} note={T('Qwen3-8B numbers: native window 32,768 tokens, 131,072 with YaRN; K/V cache 147,456 bytes per token at BF16. Prompt and answer share the window.', 'Čísla Qwen3-8B: natívne okno 32 768 tokenov, 131 072 s YaRN; cache K/V 147 456 bajtov na token pri BF16. Zadanie a odpoveď sa delia o okno.')}>
      <div className="row">
        <Range label={T('prompt tokens (input)', 'tokeny zadania (vstup)')} value={prompt} min={0} max={140000} step={500} onChange={setPrompt} fmt={(v) => num(v)} />
        <Range label={T('max output tokens', 'max. tokenov výstupu')} value={out} min={0} max={40000} step={256} onChange={setOut} fmt={(v) => num(v)} />
      </div>
      <label className="field inline"><input type="checkbox" checked={yarn} onChange={(e) => setYarn(e.target.checked)} /> {T('enable YaRN (factor 4)', 'zapnúť YaRN (faktor 4)')}</label>
      <svg viewBox={`0 0 ${W} 46`} style={{ width: '100%', maxWidth: 520, marginTop: 8 }} role="img" aria-label={T(`window ${win} tokens: prompt ${prompt}, output ${out}`, `okno ${win} tokenov: zadanie ${prompt}, výstup ${out}`)}>
        <rect x={0} y={6} width={W} height={24} rx={4} fill="var(--bg-2)" stroke="var(--line)" />
        <rect x={0} y={6} width={px(prompt)} height={24} rx={4} fill="var(--attn)" />
        <rect x={px(prompt)} y={6} width={Math.max(0, px(total) - px(prompt))} height={24} fill="var(--router)" />
        <text x={2} y={44} fontSize={9} fill="var(--text-3)">0</text>
        <text x={W - 2} y={44} fontSize={9} fill="var(--text-3)" textAnchor="end">{num(win)}</text>
      </svg>
      <p className="faint" style={{ fontSize: '0.82rem' }}><span style={{ color: 'var(--attn)' }}>■</span> {T('input', 'vstup')} · <span style={{ color: 'var(--router)' }}>■</span> {T('output', 'výstup')}</p>
      <p className="mono" style={{ fontSize: '0.88rem' }} role="status">
        {T('total', 'spolu')} {num(total)} / {num(win)} {T('tokens', 'tokenov')} ({fnum((total / win) * 100, 1)} %) · {T('room left for the answer', 'miesto na odpoveď')}: {num(Math.max(0, win - prompt))}<br />
        {T('K/V cache at the end', 'cache K/V na konci')}: <strong>{fnum(kv / 2 ** 30, 2)} GiB</strong><br />
        {over ? (prompt > win ? T('✗ The prompt alone does not fit: the request is rejected or old messages must be cut.', '✗ Samotné zadanie sa nezmestí: požiadavka sa odmietne alebo treba odrezať staré správy.') : T('⚠ The answer would be cut off when the window is full.', '⚠ Odpoveď by sa odsekla, keď sa okno zaplní.')) : yarn && total <= 32768 ? T('Fits even without YaRN; the card advises enabling YaRN only for long contexts.', 'Zmestí sa aj bez YaRN; karta odporúča zapínať YaRN len pri dlhých kontextoch.') : T('✓ Fits.', '✓ Zmestí sa.')}
      </p>
    </Frame>
  );
}

export const REGISTRY2: Record<WidgetId2, () => ReactElement> = {
  context: ContextBudget,
  dot2d: Dot2d, xent: Xent, rope2d: Rope2d, gqa: Gqa, kvcost: KvCost, journey: Journey, backprop: Backprop, quant: Quant,
};
