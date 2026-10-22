import { useEffect, useMemo, useRef, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Evidence, Section, Select, Slider, VectorView } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { CORE_PROFILES, type CoreProfile } from '../models/profiles';
import { accounting } from '../models/shapes';
import { CACHE_TOLERANCE, decodeStep, forwardFull, type KvCache } from '../math/toy';
import { maxAbsDiff } from '../math/linalg';
import { explorerHref } from '../state/route';
import { useCatalog } from '../data/data';
import { kvBytesPerToken } from '../math/memory';
import { useLang } from '../i18n/i18n';

const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export function LayersLab() {
  const { profile, trace, state } = useToy();
  const { lang, fnum } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const [pos, setPos] = useState(state.tokens.length - 1);
  const [layer, setLayer] = useState(0);
  const [real, setReal] = useState<string>('qwen3-8b');
  const P = trace[Math.min(pos, trace.length - 1)];
  const L = P.layers[Math.min(layer, profile.layers - 1)];
  const R = CORE_PROFILES.find((x) => x.id === real)!;
  const branch = profile.kind === 'dense' ? 'FFN' : 'MoE';
  return (
    <>
      <Section title={T('8 · Layer traversal (toy simulator)', '8 · Prechod vrstvami (hračkový simulátor)')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <label className="field inline">{T('Position', 'Pozícia')}<select value={Math.min(pos, state.tokens.length - 1)} onChange={(e) => setPos(Number(e.target.value))}>{state.tokens.map((t, i) => <option key={i} value={i}>{i}: {tok(t)}</option>)}</select></label>
        </div>
        <div className="table-wrap" tabIndex={0} style={{ marginTop: 6 }}>
          <table className="t">
            <thead><tr><th>{T('stage', 'krok')}</th><th className="num">‖{T('vector', 'vektor')}‖</th><th>{T('open', 'otvoriť')}</th></tr></thead>
            <tbody>
              <tr><td>{T(`embedding row for “${tok(P.token)}”`, `riadok embeddingu pre „${tok(P.token)}“`)}</td><td className="num">{fnum(norm(P.embed))}</td><td><a href={explorerHref({ m: profile.id, lv: 3, mod: 'embedding' })}>{T('module', 'modul')}</a></td></tr>
              {P.layers.map((Lx, l) => (
                <tr key={l} style={{ background: l === layer ? 'color-mix(in srgb, var(--focus) 10%, transparent)' : undefined }}>
                  <td><button className="small ghost" onClick={() => setLayer(l)}>{T('block', 'blok')} {l}</button> x → +{T('attn', 'pozor.')} ({fnum(norm(Lx.attn.out), 3)}) → +{branch} ({fnum(norm(Lx.ffnOut), 3)})</td>
                  <td className="num">{fnum(norm(Lx.out))}</td>
                  <td><a href={explorerHref({ m: profile.id, lv: 4, b: l })}>{T('block', 'blok')}</a></td>
                </tr>
              ))}
              <tr><td>{T('final RMSNorm', 'záverečná RMSNorm')}</td><td className="num">{fnum(norm(P.finalNorm))}</td><td><a href={explorerHref({ m: profile.id, lv: 3, mod: 'final_norm' })}>{T('module', 'modul')}</a></td></tr>
            </tbody>
          </table>
        </div>
        <h3>{T(`Residual additions in block ${layer} at position ${P.pos}: the values being added`, `Reziduálne sčítania v bloku ${layer} na pozícii ${P.pos}: sčítavané hodnoty`)}</h3>
        <div className="grid2">
          <div>
            <VectorView label={T('x (block input)', 'x (vstup bloku)')} values={L.xIn} />
            <VectorView label={T('+ attention branch output', '+ výstup vetvy pozornosti')} values={L.attn.out} />
            <VectorView label="= h" values={L.h} />
          </div>
          <div>
            <VectorView label="h" values={L.h} />
            <VectorView label={T(`+ ${branch} branch output`, `+ výstup vetvy ${branch}`)} values={L.ffnOut} />
            <VectorView label={T('= y (block output)', '= y (výstup bloku)')} values={L.out} />
          </div>
        </div>
        <p className="faint" style={{ fontSize: '0.86rem' }}>{T('The additions themselves have no learned weights: they simply add two activation vectors elementwise. Each block has its own matrices; the same block\'s weights are reused at every position.', 'Samotné sčítania nemajú naučené váhy: len sčítajú dva vektory aktivácií prvok po prvku. Každý blok má vlastné matice; váhy toho istého bloku sa použijú na každej pozícii.')}</p>
      </Section>
      <Section title={T('Every block of the reference models', 'Všetky bloky referenčných modelov')} badges={<Evidence kind="verified" />}>
        <Select label={T('Model', 'Model')} value={real} onChange={setReal} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <BlockTable p={R} />
      </Section>
    </>
  );
}

function BlockTable({ p }: { p: CoreProfile }) {
  const cat = useCatalog(p.id);
  const a = accounting(p);
  const { lang, num } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  return (
    <div className="table-wrap" tabIndex={0} style={{ maxHeight: 420, overflowY: 'auto', marginTop: 8 }}>
      <table className="t">
        <thead><tr><th>{T('block', 'blok')}</th><th>{T('prefix', 'predpona')}</th><th className="num">{T('params', 'parametre')}</th><th>{T('checkpoint shard(s)', 'časti checkpointu')}</th><th>{T('open', 'otvoriť')}</th></tr></thead>
        <tbody>
          {Array.from({ length: p.layers }, (_, b) => (
            <tr key={b}>
              <td>{b}</td><td className="mono">model.layers.{b}.*</td><td className="num">{num(a.perBlock)}</td>
              <td className="mono" style={{ fontSize: '0.8rem' }}>{cat.status === 'ok' ? (cat.data.layerShards[String(b)] ?? []).map((n) => `${String(n).padStart(5, '0')}`).join(', ') + (p.kind === 'moe' && cat.data.expertShardPlacement ? ` · ${T('experts in', 'experti v')} ${[...new Set([...cat.data.expertShardPlacement[String(b)].gate, ...cat.data.expertShardPlacement[String(b)].down])].join(', ')}` : '') : cat.status === 'error' ? T('catalog unavailable', 'katalóg nedostupný') : '…'}</td>
              <td><a href={explorerHref({ m: p.id, lv: 4, b })}>{T('block', 'blok')} {b}</a></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface Step { kind: 'prefill' | 'decode'; layer?: number; pos?: number; label: string }

export function CacheLab() {
  const { profile: p, weights, state, dispatch, stateKey } = useToy();
  const { lang, bytes, big } = useLang();
  const T = (en: string, sk: string) => (lang === 'sk' ? sk : en);
  const exp = (x: number) => (lang === 'sk' ? x.toExponential(2).replace('.', ',') : x.toExponential(2));
  const prompt = state.tokens;
  const nDecode = 4;
  // Build the prefill (layer-major, all positions) and greedy decode schedule.
  const run = useMemo(() => {
    const pre = forwardFull(p, weights, prompt);
    const cache0: KvCache = { stateKey, tokens: prompt.slice(), layers: Array.from({ length: p.layers }, (_, l) => ({ k: pre.map((t) => t.layers[l].attn.kRot), v: pre.map((t) => t.layers[l].attn.v) })) };
    const decodes: { token: number; pos: number; cachedLogits: number[]; fullLogits: number[]; diff: number; cache: KvCache }[] = [];
    let cache = cache0;
    let next = argmax(pre[pre.length - 1].logits);
    for (let i = 0; i < nDecode; i++) {
      const r = decodeStep(p, weights, cache, next);
      cache = r.cache;
      const full = forwardFull(p, weights, cache.tokens);
      const fl = full[full.length - 1].logits;
      decodes.push({ token: next, pos: r.trace.pos, cachedLogits: r.trace.logits, fullLogits: fl, diff: maxAbsDiff(r.trace.logits, fl), cache });
      next = argmax(r.trace.logits);
    }
    return { pre, cache0, decodes };
  }, [p, weights, prompt, stateKey]);
  const steps: Step[] = [
    ...Array.from({ length: p.layers }, (_, l) => ({ kind: 'prefill' as const, layer: l, label: T(`Prefill · block ${l} (positions 0–${prompt.length - 1} together)`, `Predvyplnenie · blok ${l} (pozície 0–${prompt.length - 1} naraz)`) })),
    ...run.decodes.map((d, i) => ({ kind: 'decode' as const, pos: d.pos, label: T(`Decode ${i + 1} · “${tok(d.token)}” at position ${d.pos}`, `Generovanie ${i + 1} · „${tok(d.token)}“ na pozícii ${d.pos}`) })),
  ];
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => setK(0), [stateKey, prompt.join(',')]);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setTimeout(() => (k < steps.length - 1 ? setK(k + 1) : setPlaying(false)), 1400 / speed);
    return () => window.clearTimeout(timer.current);
  }, [playing, k, speed, steps.length]);
  const cur = steps[Math.min(k, steps.length - 1)];
  const cacheLen = (l: number) => (cur.kind === 'prefill' ? (l <= cur.layer! ? prompt.length : 0) : run.decodes[steps.indexOf(cur) - p.layers].cache.layers[l].k.length);
  const allOk = run.decodes.every((d) => d.diff <= CACHE_TOLERANCE);
  const [syncInfo, setSyncInfo] = useState<string | null>(null);
  useEffect(() => { if (state.lastSync) setSyncInfo(state.log[0] ?? null); }, [state.lastSync]); // eslint-disable-line react-hooks/exhaustive-deps
  const storeFull = useMemo(() => forwardFull(p, weights, state.tokens), [p, weights, state.tokens]);
  const prev = cur.kind === 'prefill' ? (cur.layer! - 1 >= 0 ? String(cur.layer! - 1) : T('the embedding', 'embeddingu')) : '';
  return (
    <>
      <Section title={T('9 · Prefill, decode and the K/V cache', '9 · Predvyplnenie, generovanie a cache K/V')} badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row" role="toolbar" aria-label={T('Timeline controls', 'Ovládanie časovej osi')} style={{ margin: '8px 0' }}>
          <button onClick={() => setK(Math.max(0, k - 1))} disabled={k === 0}>◀ {T('Previous', 'Predchádzajúci')}</button>
          <button onClick={() => setPlaying(!playing)} aria-pressed={playing}>{playing ? `❚❚ ${T('Pause', 'Pozastaviť')}` : `▶ ${T('Play', 'Prehrať')}`}</button>
          <button onClick={() => setK(Math.min(steps.length - 1, k + 1))} disabled={k >= steps.length - 1}>{T('Next', 'Ďalší')} ▶</button>
          <Slider label={T('speed', 'rýchlosť')} value={speed} min={0.5} max={3} step={0.5} onChange={setSpeed} format={(v) => `${lang === 'sk' ? String(v).replace('.', ',') : v}×`} />
          {reduce && <span className="faint" style={{ fontSize: '0.8rem' }}>{T('Reduced motion: no animation; playback only advances steps.', 'Obmedzený pohyb: bez animácie; prehrávanie len posúva kroky.')}</span>}
        </div>
        <div className="timeline" aria-label={T('Steps', 'Kroky')}>
          {steps.map((s, i) => <button key={i} className={`step${i === k ? ' cur' : i < k ? ' done' : ''}`} onClick={() => setK(i)} aria-current={i === k ? 'step' : undefined}>{s.label}</button>)}
        </div>
        <div className="grid2" style={{ marginTop: 10 }}>
          <div>
            <h3 aria-live="polite">{cur.label}</h3>
            {cur.kind === 'prefill' ? (
              <p>{T(
                `All ${prompt.length} prompt positions pass through block ${cur.layer} at once (parallel across positions, causal mask inside attention). Block ${cur.layer} must wait for block ${prev} to finish: blocks are a sequential dependency. Block ${cur.layer} stores K and V for every prompt position.`,
                `Všetkých ${prompt.length} pozícií zadania prejde blokom ${cur.layer} naraz (paralelne cez pozície, s kauzálnou maskou v pozornosti). Blok ${cur.layer} musí počkať, kým skončí ${cur.layer! - 1 >= 0 ? `blok ${prev}` : prev}: bloky na sebe postupne závisia. Blok ${cur.layer} uloží K a V pre každú pozíciu zadania.`,
              )}</p>
            ) : lang === 'sk' ? (
              <p>Spracuje sa len nový token „{tok(run.decodes[k - p.layers].token)}“, a to všetkými {p.layers} blokmi. Jeho dopyt číta uložené K/V pozícií 0–{cur.pos! - 1} plus svoje vlastné; jeho K/V sa pripoja. RoPE použije jeho absolútnu pozíciu <strong>{cur.pos}</strong> (= dĺžka cache pred krokom). Staršie K/V sa znovu použijú, neprepočítavajú sa. Ďalší token sa vyberie hladovo z hračkových logitov.</p>
            ) : (
              <p>Only the new token “{tok(run.decodes[k - p.layers].token)}” is processed, through all {p.layers} blocks. Its query reads the cached K/V of positions 0–{cur.pos! - 1} plus its own; its K/V are appended. RoPE uses its absolute position <strong>{cur.pos}</strong> (= cache length before the step). Earlier K/V are reused, not recomputed. Next token chosen greedily from the toy logits.</p>
            )}
            <p className="faint" style={{ fontSize: '0.86rem' }}>{T('The cache is computed runtime state, not trained parameters. It is cleared whenever the weights, seed, configuration or an earlier token change.', 'Cache je vypočítaný stav počas behu, nie naučené parametre. Vymaže sa vždy, keď sa zmenia váhy, osivo, konfigurácia alebo niektorý skorší token.')}</p>
          </div>
          <div>
            <h3>{T('Cache contents', 'Obsah cache')}</h3>
            <table className="t"><thead><tr><th>{T('block', 'blok')}</th><th>{T('K shape [Hkv, T, d_h]', 'tvar K [Hkv, T, d_h]')}</th><th>{T('V shape', 'tvar V')}</th><th>{T('positions', 'pozície')}</th></tr></thead>
              <tbody>{Array.from({ length: p.layers }, (_, l) => {
                const n = cacheLen(l);
                return <tr key={l}><td>{l}</td><td className="mono">[{p.hkv}, {n}, {p.headDim}]</td><td className="mono">[{p.hkv}, {n}, {p.headDim}]</td><td className="mono">{n ? `0–${n - 1}` : T('empty', 'prázdne')}{cur.kind === 'decode' && n ? T(` (new: ${n - 1})`, ` (nová: ${n - 1})`) : ''}</td></tr>;
              })}</tbody>
            </table>
            <p className="faint" style={{ fontSize: '0.84rem' }}>{T(`Toy: ${p.hkv} K/V head × ${p.headDim} values × 2 (K and V) per position per block. Real: `, `Hračka: ${p.hkv} hlava K/V × ${p.headDim} hodnôt × 2 (K a V) na pozíciu a blok. Skutočné: `)}{CORE_PROFILES.map((x) => `${x.name} ${bytes(kvBytesPerToken(x))}/${T('token', 'token')}`).join(' · ')}{T(' at BF16.', ' pri BF16.')}</p>
          </div>
        </div>
      </Section>
      <Section title={T('Correctness check: cached decoding = full-prefix recomputation', 'Kontrola správnosti: generovanie s cache = prepočet celej predpony')} badges={<Evidence kind="simulation" />}>
        <p>{lang === 'sk'
          ? <>Dve nezávislé cesty v kóde: <code>forwardFull</code> (vrstva po vrstve cez všetky pozície s explicitnou kauzálnou maskou) a <code>decodeStep</code> (jeden token všetkými blokmi s čítaním z cache). Tolerancia {CACHE_TOLERANCE}.</>
          : <>Two independent code paths: <code>forwardFull</code> (layer-major over every position with an explicit causal mask) and <code>decodeStep</code> (one token through all blocks, reading the cache). Tolerance {CACHE_TOLERANCE}.</>}</p>
        <table className="t"><thead><tr><th>{T('step', 'krok')}</th><th>{T('token', 'token')}</th><th>{T('position', 'pozícia')}</th><th className="num">{T('max |logit difference|', 'max. |rozdiel logitov|')}</th><th>{T('result', 'výsledok')}</th></tr></thead>
          <tbody>{run.decodes.map((d, i) => <tr key={i}><td>{T('decode', 'generovanie')} {i + 1}</td><td className="mono">{tok(d.token)}</td><td>{d.pos}</td><td className="num">{exp(d.diff)}</td><td>{d.diff <= CACHE_TOLERANCE ? T('✓ equal within tolerance', '✓ zhodné v rámci tolerancie') : T('✗ MISMATCH', '✗ NEZHODA')}</td></tr>)}</tbody>
        </table>
        <p><strong>{allOk ? T('✓ All decode steps match.', '✓ Všetky kroky generovania sa zhodujú.') : T('✗ A mismatch was found.', '✗ Našla sa nezhoda.')}</strong></p>
      </Section>
      <Section title={T('Cache invalidation (shared simulator cache)', 'Zneplatnenie cache (spoločná cache simulátora)')} badges={<Evidence kind="simulation" />}>
        <p className="muted">{T('The shared simulator keeps a real K/V cache keyed by its model state. Edit a prompt token above, change the seed, or edit a weight in the Weight lab, then synchronize: incompatible entries are dropped before recomputation.', 'Spoločný simulátor si drží skutočnú cache K/V viazanú na stav modelu. Upravte token zadania vyššie, zmeňte osivo alebo upravte váhu v laboratóriu váh a potom synchronizujte: nekompatibilné položky sa pred prepočtom zahodia.')}</p>
        <div className="row">
          <button className="primary" onClick={() => dispatch({ type: 'sync' })}>{T('Synchronize cache with current prompt', 'Synchronizovať cache s aktuálnym zadaním')}</button>
          <button onClick={() => dispatch({ type: 'clearCache' })}>{T('Clear cache', 'Vymazať cache')}</button>
        </div>
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>{T('Cached positions', 'Pozície v cache')}</dt><dd>{state.cache.tokens.length} ({state.cache.tokens.map(tok).join(' ') || T('empty', 'prázdne')})</dd>
          <dt>{T('Cache state key', 'Kľúč stavu cache')}</dt><dd className="mono" style={{ fontSize: '0.78rem' }}>{state.cache.stateKey === stateKey ? T('matches current model state', 'zodpovedá aktuálnemu stavu modelu') : T('STALE: model state changed since it was built', 'ZASTARANÁ: stav modelu sa od jej vytvorenia zmenil')}</dd>
          <dt>{T('Prefix match', 'Zhoda predpony')}</dt><dd>{(() => { let i = 0; while (i < state.cache.tokens.length && state.cache.tokens[i] === state.tokens[i]) i++; return i === state.cache.tokens.length ? T(`cache is a prefix of the prompt (${state.tokens.length - i} position(s) still to decode)`, `cache je predponou zadania (zostáva spracovať ${state.tokens.length - i} pozícií)`) : T(`prompt differs from cache at position ${i}: positions ${i}+ are invalid`, `zadanie sa od cache líši na pozícii ${i}: pozície od ${i} sú neplatné`); })()}</dd>
          {syncInfo && <><dt>{T('Last action', 'Posledná akcia')}</dt><dd>{syncInfo}</dd></>}
          {state.lastSync && state.cache.tokens.length === state.tokens.length && state.cache.stateKey === stateKey && (
            <><dt>{T('Check', 'Kontrola')}</dt><dd>{(() => {
              const last = decodeStepLogits(p, weights, state.cache);
              const diff = maxAbsDiff(last, storeFull[storeFull.length - 1].logits);
              return `${diff <= CACHE_TOLERANCE ? '✓' : '✗'} ${T(`last-position logits from the cache path vs full recomputation: max difference ${exp(diff)}`, `logity poslednej pozície z cesty s cache vs úplný prepočet: max. rozdiel ${exp(diff)}`)}`;
            })()}</dd></>
          )}
        </dl>
        <p className="faint" style={{ fontSize: '0.84rem' }}>{T('Inspection never touches this cache: selecting heads, positions or steps above is read-only.', 'Prezeranie sa tejto cache nikdy nedotkne: výber hláv, pozícií či krokov vyššie iba číta.')}</p>
      </Section>
      <Section title={T('Newer hybrid models need a different state accounting', 'Novšie hybridné modely potrebujú iné počítanie stavu')} badges={<Evidence kind="derived" />}>
        {lang === 'sk'
          ? <p>Bloky Gated DeltaNet si namiesto rastúcej cache K/V držia rekurentný stav pevnej veľkosti pre každú sekvenciu a n-gramové tabuľky modelu Qwen3.8-Flash-Next sú naučené parametre, nie stav. Jeho bloky pozornosti si stále držia kľúče a hodnoty pre každý token. Bežný vzorec vyššie preto neplatí; pozrite <a href="#/hybrid/qwen3.8-flash-next">stránku Qwen3.8-Flash-Next</a> a <a href="#/labs/memory">laboratórium pamäte</a>.</p>
          : <p>Gated DeltaNet blocks keep a fixed-size recurrent state per sequence instead of a growing K/V cache, and Qwen3.8-Flash-Next's n-gram tables are learned parameters, not state. Its attention blocks still keep per-token keys and values. The ordinary formula above therefore does not apply; see the <a href="#/hybrid/qwen3.8-flash-next">Qwen3.8-Flash-Next page</a> and the <a href="#/labs/memory">memory lab</a>.</p>}
        <p className="faint">{CORE_PROFILES.map((x) => `${x.name}: ${big(accounting(x).total)} ${T('parameters', 'parametrov')}`).join(' · ')}</p>
      </Section>
    </>
  );
}

function argmax(v: number[]) { return v.reduce((b, x, i) => (x > v[b] ? i : b), 0); }

/** Re-derive the last position's logits through the cached path: recompute the final token against the cache minus itself. */
function decodeStepLogits(p: CoreProfile, w: Parameters<typeof decodeStep>[1], cache: KvCache) {
  const n = cache.tokens.length;
  const prefix: KvCache = { stateKey: cache.stateKey, tokens: cache.tokens.slice(0, n - 1), layers: cache.layers.map((L) => ({ k: L.k.slice(0, n - 1), v: L.v.slice(0, n - 1) })) };
  return decodeStep(p, w, prefix, cache.tokens[n - 1]).trace.logits;
}
