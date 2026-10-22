import { useState } from 'react';
import { Cite, Evidence, NumberField, Section, Select, Slider, Tex } from '../components/ui';
import { CORE_PROFILES, TOY_DENSE } from '../models/profiles';
import { accounting, attentionOpsPerToken, matmulOpsPerToken } from '../models/shapes';
import { GB, GiB, hybridStateBytes, kvCacheBytes, quantizedEstimate, trainingMemory, weightBytes } from '../math/memory';
import { QWEN38_COMPONENTS, QWEN38_STATE } from '../models/hybrid';
import { forwardFull, initToyWeights, DEFAULT_PROMPT } from '../math/toy';
import { useLang } from '../i18n/i18n';

export function MemoryLab() {
  const { lang, num: fmtInt, bytes: fmtBytes, big: fmtB } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const dec = (s: string) => (lang === 'sk' ? s.replace('.', ',') : s);
  const [mid, setMid] = useState(CORE_PROFILES[0].id);
  const p = CORE_PROFILES.find((x) => x.id === mid)!;
  const a = accounting(p);
  const [bits, setBits] = useState(16);
  const [group, setGroup] = useState(64);
  const [scaleBits, setScaleBits] = useState(16);
  const [zp, setZp] = useState(0);
  const [keepEmb, setKeepEmb] = useState(true);
  const [batch, setBatch] = useState(1);
  const [ctx, setCtx] = useState(32768);
  const [kvBytes, setKvBytes] = useState(2);
  const raw = weightBytes(a.total, bits);
  const q = quantizedEstimate(p, { bits, groupSize: group, scaleBits, zeroPointBits: zp, keepUnquantized: { embedding: keepEmb, head: keepEmb, norms: true }, unquantizedBits: 16 });
  const kv = kvCacheBytes(p, { batch, tokens: ctx, bytesPerScalar: kvBytes });
  const tm = trainingMemory(a.total, { weightBytes: 2, gradBytes: 2, adamStateBytes: 4, masterCopyBytes: 4 });
  const [measure, setMeasure] = useState<{ ms: number; runs: number } | null>(null);
  const hybrid = hybridStateBytes(QWEN38_STATE, { tokens: ctx, batch, kvBytes, stateBytes: 4 });
  const q38total = QWEN38_COMPONENTS.reduce((s, c) => s + c.elements, 0);
  return (
    <>
      <Section title={T('12 · Memory lab', '12 · Laboratórium pamäte')} badges={<><Evidence kind="derived" /></>}>
        <Select label={T('Model', 'Model')} value={mid} onChange={setMid} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <div className="grid2" style={{ marginTop: 10 }}>
          <div className="card">
            <h3>{T('Weights', 'Váhy')}</h3>
            <Tex block>{lang === 'sk' ? '\\text{bajty váh} = N_{param} \\times \\text{bity} / 8' : '\\text{weight bytes} = N_{params} \\times \\text{bits} / 8'}</Tex>
            <div className="row">
              <Select label={T('bits per weight', 'bity na váhu')} value={bits} onChange={setBits} options={[{ value: 32, label: '32 (FP32)' }, { value: 16, label: T('16 (BF16 — as published)', '16 (BF16 — ako bol zverejnený)') }, { value: 8, label: '8 (FP8 / INT8)' }, { value: 4, label: T('4 (4-bit)', '4 (4-bitové)') }]} />
            </div>
            <p className="mono">{fmtInt(a.total)} × {bits} / 8 = <strong>{fmtBytes(raw)}</strong></p>
            {bits < 16 && (
              <>
                <p className="faint" style={{ fontSize: '0.86rem' }}>{T('Ideal payload only. Real quantized formats add scales (and sometimes zero-points) per group, often keep some tensors at higher precision, and have container overhead. Quantization changes how numbers are stored, not how many parameters the architecture has.', 'Len ideálny objem dát. Skutočné kvantizované formáty pridávajú mierky (a niekedy nulové body) pre každú skupinu, niektoré tenzory často ponechajú vo vyššej presnosti a majú réžiu kontajnera. Kvantizácia mení spôsob uloženia čísel, nie počet parametrov architektúry.')}</p>
                <div className="row">
                  <NumberField label={T('group size', 'veľkosť skupiny')} value={group} min={8} step={8} width={90} onChange={(v) => setGroup(Math.max(8, Math.round(v)))} />
                  <Select label={T('scale bits', 'bity mierky')} value={scaleBits} onChange={setScaleBits} options={[{ value: 16, label: '16' }, { value: 32, label: '32' }, { value: 8, label: '8' }]} />
                  <Select label={T('zero-point bits', 'bity nulového bodu')} value={zp} onChange={setZp} options={[{ value: 0, label: T('0 (symmetric)', '0 (symetrická)') }, { value: bits, label: T(`${bits} (asymmetric)`, `${bits} (asymetrická)`) }]} />
                  <label className="field inline"><input type="checkbox" checked={keepEmb} onChange={(e) => setKeepEmb(e.target.checked)} /> {T('keep embedding + head at 16-bit', 'ponechať embedding + hlavu v 16 bitoch')}</label>
                </div>
                <dl className="kv" style={{ marginTop: 6 }}>
                  <dt>{T('quantized payload', 'kvantizované dáta')}</dt><dd>{fmtBytes(q.payloadBytes)} ({fmtInt(q.quantizedParams)} {T('params', 'parametrov')})</dd>
                  <dt>{T('scale/zero metadata', 'metadáta mierok/nulových bodov')}</dt><dd>{fmtBytes(q.metadataBytes)}</dd>
                  <dt>{T('unquantized tensors', 'nekvantizované tenzory')}</dt><dd>{fmtBytes(q.unquantizedBytes)} ({fmtInt(q.unquantizedParams)} {T('params at 16-bit', 'parametrov v 16 bitoch')})</dd>
                  <dt>{T('estimate', 'odhad')}</dt><dd><strong>{fmtBytes(q.totalBytes)}</strong> {T('— not a guaranteed GGUF or Safetensors file size', '— nie je to zaručená veľkosť súboru GGUF ani Safetensors')}</dd>
                </dl>
                <p className="faint" style={{ fontSize: '0.82rem' }}>{T('GGUF is a container format; one GGUF file can hold many different quantization types.', 'GGUF je formát kontajnera; jeden súbor GGUF môže obsahovať mnoho rôznych typov kvantizácie.')}<Cite ids={['gguf']} /></p>
              </>
            )}
            {p.kind === 'moe' && <div className="callout">{T(`Only ~3.3B parameters are “activated” per token, but all ${fmtB(a.total)} must be stored and reachable, because different tokens and blocks pick different experts.`, `Na token je „aktivovaných“ len asi 3,3 mld. parametrov, ale všetkých ${fmtB(a.total)} musí byť uložených a dostupných, lebo rôzne tokeny a bloky vyberajú rôznych expertov.`)}</div>}
          </div>
          <div className="card">
            <h3>{T('K/V cache (ordinary full-attention Qwen3 blocks)', 'Cache K/V (bežné bloky Qwen3 s plnou pozornosťou)')}</h3>
            <Tex block>{lang === 'sk' ? '\\text{bajty KV} = 2 \\cdot B \\cdot L \\cdot T \\cdot H_{kv} \\cdot d_h \\cdot \\text{bajty}' : '\\text{KV bytes} = 2 \\cdot B \\cdot L \\cdot T \\cdot H_{kv} \\cdot d_h \\cdot \\text{bytes}'}</Tex>
            <div className="row">
              <Slider label={T('batch B', 'dávka B')} value={batch} min={1} max={64} onChange={setBatch} />
              <Select label={T('context T', 'kontext T')} value={ctx} onChange={setCtx} options={[1024, 4096, 8192, 32768, 40960, 131072].map((v) => ({ value: v, label: fmtInt(v) }))} />
              <Select label={T('cache precision', 'presnosť cache')} value={kvBytes} onChange={setKvBytes} options={[{ value: 2, label: 'BF16 / FP16 (2 B)' }, { value: 1, label: 'FP8 (1 B)' }, { value: 4, label: 'FP32 (4 B)' }]} />
            </div>
            <p className="mono">2 × {batch} × {p.layers} × {fmtInt(ctx)} × {p.hkv} × {p.headDim} × {kvBytes} = <strong>{fmtBytes(kv)}</strong></p>
            <p className="faint" style={{ fontSize: '0.86rem' }}>{T(
              `K/V only: excludes allocator overhead, paging granularity, activations and kernel workspaces. Uses K/V heads (${p.hkv}), not query heads (${p.hq}). Weight precision and cache precision are independent settings.`,
              `Len K/V: bez réžie alokátora, granularity stránkovania, aktivácií a pracovnej pamäte jadier. Používa hlavy K/V (${p.hkv}), nie hlavy dopytov (${p.hq}). Presnosť váh a presnosť cache sú nezávislé nastavenia.`,
            )}{ctx > p.context.native && T(` ${fmtInt(ctx)} exceeds the native ${fmtInt(p.context.native)}; the model card describes YaRN (factor 4) for up to ${fmtInt(p.context.yarn!)}.`, ` ${fmtInt(ctx)} presahuje natívnych ${fmtInt(p.context.native)}; karta modelu opisuje YaRN (faktor 4) až do ${fmtInt(p.context.yarn!)}.`)}</p>
            <h3>{T('Training (persistent state only)', 'Trénovanie (len trvalý stav)')}</h3>
            <p className="mono" style={{ fontSize: '0.86rem' }}>{T('BF16 weights', 'váhy BF16')} {fmtBytes(tm.weights)} + {T('BF16 grads', 'gradienty BF16')} {fmtBytes(tm.gradients)} + {T('Adam m, v in FP32', 'Adam m, v v FP32')} {fmtBytes(tm.optimizerStates)} + {T('FP32 master copy', 'hlavná kópia FP32')} {fmtBytes(tm.masterWeights)} = <strong>{fmtBytes(tm.total)}</strong></p>
            <p className="faint" style={{ fontSize: '0.84rem' }}>{T('One common mixed-precision layout (16 bytes/parameter); others exist. Activation memory for backpropagation depends on batch, sequence length and recomputation strategy and is not included.', 'Jedno bežné rozloženie so zmiešanou presnosťou (16 bajtov na parameter); existujú aj iné. Pamäť aktivácií pre spätné šírenie závisí od dávky, dĺžky sekvencie a stratégie prepočtu a nie je zahrnutá.')}</p>
          </div>
        </div>
      </Section>
      <Section title={T('Compute, traffic and what is architecture vs serving choice', 'Výpočty, prenosy a čo je architektúra a čo voľba prevádzky')} badges={<Evidence kind="derived" />}>
        <div className="table-wrap" tabIndex={0}><table className="t"><tbody>
          <tr><td>{T('Matrix-multiply ops per generated token (2 × active matmul params, excl. attention scores)', 'Operácie násobenia matíc na generovaný token (2 × aktívne parametre matíc, bez skóre pozornosti)')}</td><td className="num">{fmtB(matmulOpsPerToken(p))}</td></tr>
          <tr><td>{T(`Attention score + value ops for one token over T = ${fmtInt(ctx)} earlier tokens`, `Operácie skóre a hodnôt pozornosti pre jeden token cez T = ${fmtInt(ctx)} skorších tokenov`)}</td><td className="num">{fmtB(attentionOpsPerToken(p, ctx))}</td></tr>
          <tr><td>{T('Weight bytes that must be read per decode step (BF16, batch 1, dense reading of active weights)', 'Bajty váh, ktoré treba prečítať v kroku generovania (BF16, dávka 1, husté čítanie aktívnych váh)')}</td><td className="num">{p.kind === 'dense' ? fmtBytes(a.total * 2) : `${fmtBytes((a.activeEmbeddingRowOnly!) * 2)} ${T('(selected experts only)', '(len vybraní experti)')}`}</td></tr>
        </tbody></table></div>
        {lang === 'sk' ? (
          <ul style={{ paddingLeft: 18, fontSize: '0.92rem' }}>
            <li>Počty parametrov, počty operácií, prenosy pamäte, nameraná latencia a využitie hardvéru sú rôzne veličiny; táto stránka ukazuje len prvé dve a odhad prenosov. Neuvádza sa žiadny počet tokenov za sekundu.</li>
            <li><strong>Dlaždice a akumulácia.</strong> Jadrá GPU delia matice na dlaždice, ktoré sa zmestia do rýchlej pamäte čipu, a súčiny BF16 zvyčajne sčítavajú v FP32. FlashAttention používa tú istú myšlienku pre pozornosť s priebežným softmaxom.<Cite ids={['flashattn']} /></li>
            <li><strong>Paralelizmus.</strong> Tenzorový paralelizmus delí jednotlivé matice medzi zariadenia; potrubný paralelizmus delí bloky; paralelizmus expertov umiestňuje expertov na rôzne zariadenia a presúva k nim aktivácie tokenov (všetci so všetkými).<Cite ids={['switch']} /> Sú to voľby implementácie prevádzky a trénovania, nie súčasť architektúry modelu.</li>
          </ul>
        ) : (
          <ul style={{ paddingLeft: 18, fontSize: '0.92rem' }}>
            <li>Parameter counts, operation counts, memory traffic, measured latency and hardware utilization are different quantities; this page shows only the first two plus a traffic estimate. No tokens-per-second figure is claimed.</li>
            <li><strong>Tiling and accumulation.</strong> GPU kernels split matrices into tiles that fit fast on-chip memory and usually accumulate BF16 products in FP32. FlashAttention applies the same idea to attention with an online softmax.<Cite ids={['flashattn']} /></li>
            <li><strong>Parallelism.</strong> Tensor parallelism splits individual matrices across devices; pipeline parallelism splits blocks; expert parallelism places experts on different devices and moves token activations to them (all-to-all).<Cite ids={['switch']} /> These are serving/training implementation choices, not part of the model architecture.</li>
          </ul>
        )}
        <div className="row">
          <button onClick={() => {
            const w = initToyWeights(TOY_DENSE, 1);
            const runs = 200;
            const t0 = performance.now();
            for (let i = 0; i < runs; i++) forwardFull(TOY_DENSE, w, DEFAULT_PROMPT);
            setMeasure({ ms: (performance.now() - t0) / runs, runs });
          }}>{T('Measure the toy simulator in this browser', 'Zmerať hračkový simulátor v tomto prehliadači')}</button>
        </div>
        {measure && <p className="callout" style={{ fontSize: '0.86rem' }}>{T(
          `Measured just now in this browser: toy dense forward pass over ${DEFAULT_PROMPT.length} tokens took ${measure.ms.toFixed(3)} ms on average (${measure.runs} runs). Conditions: ${navigator.userAgent}; ${navigator.hardwareConcurrency ?? '?'} logical CPUs reported. A tiny float64 JavaScript simulator; says nothing about Qwen speed on any hardware.`,
          `Práve namerané v tomto prehliadači: hustý hračkový dopredný prechod cez ${DEFAULT_PROMPT.length} tokenov trval v priemere ${dec(measure.ms.toFixed(3))} ms (${measure.runs} behov). Podmienky: ${navigator.userAgent}; hlásených ${navigator.hardwareConcurrency ?? '?'} logických CPU. Ide o malý simulátor v JavaScripte s float64; o rýchlosti Qwen na akomkoľvek hardvéri nehovorí nič.`,
        )}</p>}
      </Section>
      <Section title={T('Hybrid model state: Qwen3.8-Flash-Next', 'Stav hybridného modelu: Qwen3.8-Flash-Next')} badges={<><Evidence kind="derived" /><Evidence kind="unknown" title="State dtype is a runtime choice" /></>}>
        <p>{T(
          'Derived from the pinned implementation and config: only the 12 QSA blocks keep per-token K/V (2 K/V heads × 256) plus 128-wide indexer keys; the 36 Gated DeltaNet blocks keep a fixed-size recurrent state [48 heads × 128 × 128] plus short convolution state; the n-gram tables (51.2B parameters) are weights, not state. Uses the context and batch above; KV precision as selected; recurrent state assumed FP32 (config mamba_ssm_dtype is float32; a runtime may differ).',
          'Odvodené z pripnutej implementácie a konfigurácie: K/V pre každý token (2 hlavy K/V × 256) a 128-rozmerné kľúče indexera si držia len 12 blokov QSA; 36 blokov Gated DeltaNet si drží rekurentný stav pevnej veľkosti [48 hláv × 128 × 128] a krátky stav konvolúcie; n-gramové tabuľky (51,2 mld. parametrov) sú váhy, nie stav. Používa kontext a dávku vyššie; presnosť KV podľa výberu; rekurentný stav predpokladá FP32 (v konfigurácii je mamba_ssm_dtype float32; prostredie behu sa môže líšiť).',
        )}</p>
        <div className="table-wrap" tabIndex={0}><table className="t"><tbody>
          <tr><td>{T('QSA K/V per token (12 blocks)', 'QSA K/V na token (12 blokov)')}</td><td className="num">{fmtInt(hybrid.kvPerToken)} B</td></tr>
          <tr><td>{T('QSA indexer keys per token', 'Kľúče indexera QSA na token')}</td><td className="num">{fmtInt(hybrid.indexerPerToken)} B</td></tr>
          <tr><td>{T(`Growing state at T = ${fmtInt(ctx)}, batch ${batch}`, `Rastúci stav pri T = ${fmtInt(ctx)}, dávka ${batch}`)}</td><td className="num">{fmtBytes(hybrid.growing)}</td></tr>
          <tr><td>{T('Fixed recurrent state (36 GDN blocks)', 'Pevný rekurentný stav (36 blokov GDN)')}</td><td className="num">{fmtBytes(hybrid.recurrent)}</td></tr>
          <tr><td>{T('Convolution states (approximate)', 'Stavy konvolúcie (približne)')}</td><td className="num">{fmtBytes(hybrid.conv)}</td></tr>
          <tr><td>{T('Stored checkpoint elements (all components)', 'Uložené prvky checkpointu (všetky časti)')}</td><td className="num">{fmtInt(q38total)} ≈ {dec((q38total * 2 / GB).toFixed(1))} GB = {dec((q38total * 2 / GiB).toFixed(1))} GiB {T('at BF16', 'pri BF16')}</td></tr>
        </tbody></table></div>
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('So total state still grows with context (attention blocks), just more slowly than if all 48 blocks were full attention; the model as a whole is not constant-memory. MTP and vision components are excluded from the per-token state above.', 'Celkový stav teda stále rastie s kontextom (bloky pozornosti), len pomalšie, než keby všetkých 48 blokov malo plnú pozornosť; model ako celok nemá konštantnú pamäť. Časti MTP a zraku nie sú v stave na token vyššie zahrnuté.')}</p>
      </Section>
    </>
  );
}
