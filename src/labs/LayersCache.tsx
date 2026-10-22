import { useEffect, useMemo, useRef, useState } from 'react';
import { useToy } from '../state/toyStore';
import { Evidence, Section, Select, Slider, VectorView, fmtB, fmtInt, fmtNum } from '../components/ui';
import { ToyControls, PromptEditor, tok } from './common';
import { CORE_PROFILES, type CoreProfile } from '../models/profiles';
import { accounting } from '../models/shapes';
import { CACHE_TOLERANCE, decodeStep, forwardFull, type KvCache } from '../math/toy';
import { maxAbsDiff } from '../math/linalg';
import { explorerHref } from '../state/route';
import { useCatalog } from '../data/data';
import { kvBytesPerToken, fmtBytes } from '../math/memory';

const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

export function LayersLab() {
  const { profile, trace, state } = useToy();
  const [pos, setPos] = useState(state.tokens.length - 1);
  const [layer, setLayer] = useState(0);
  const [real, setReal] = useState<string>('qwen3-8b');
  const P = trace[Math.min(pos, trace.length - 1)];
  const L = P.layers[Math.min(layer, profile.layers - 1)];
  const R = CORE_PROFILES.find((x) => x.id === real)!;
  return (
    <>
      <Section title="8 · Layer traversal (toy simulator)" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row">
          <label className="field inline">Position<select value={Math.min(pos, state.tokens.length - 1)} onChange={(e) => setPos(Number(e.target.value))}>{state.tokens.map((t, i) => <option key={i} value={i}>{i}: {tok(t)}</option>)}</select></label>
        </div>
        <div className="table-wrap" style={{ marginTop: 6 }}>
          <table className="t">
            <thead><tr><th>stage</th><th className="num">‖vector‖</th><th>open</th></tr></thead>
            <tbody>
              <tr><td>embedding row for “{tok(P.token)}”</td><td className="num">{fmtNum(norm(P.embed))}</td><td><a href={explorerHref({ m: profile.id, lv: 3, mod: 'embedding' })}>module</a></td></tr>
              {P.layers.map((Lx, l) => (
                <tr key={l} style={{ background: l === layer ? 'color-mix(in srgb, var(--focus) 10%, transparent)' : undefined }}>
                  <td><button className="small ghost" onClick={() => setLayer(l)}>block {l}</button> x → +attn ({fmtNum(norm(Lx.attn.out), 3)}) → +{profile.kind === 'dense' ? 'FFN' : 'MoE'} ({fmtNum(norm(Lx.ffnOut), 3)})</td>
                  <td className="num">{fmtNum(norm(Lx.out))}</td>
                  <td><a href={explorerHref({ m: profile.id, lv: 4, b: l })}>block</a></td>
                </tr>
              ))}
              <tr><td>final RMSNorm</td><td className="num">{fmtNum(norm(P.finalNorm))}</td><td><a href={explorerHref({ m: profile.id, lv: 3, mod: 'final_norm' })}>module</a></td></tr>
            </tbody>
          </table>
        </div>
        <h3>Residual additions in block {layer} at position {P.pos}: the values being added</h3>
        <div className="grid2">
          <div>
            <VectorView label="x (block input)" values={L.xIn} />
            <VectorView label="+ attention branch output" values={L.attn.out} />
            <VectorView label="= h" values={L.h} />
          </div>
          <div>
            <VectorView label="h" values={L.h} />
            <VectorView label={`+ ${profile.kind === 'dense' ? 'FFN' : 'MoE'} branch output`} values={L.ffnOut} />
            <VectorView label="= y (block output)" values={L.out} />
          </div>
        </div>
        <p className="faint" style={{ fontSize: '0.86rem' }}>The additions themselves have no learned weights: they simply add two activation vectors elementwise. Each block has its own matrices; the same block's weights are reused at every position.</p>
      </Section>
      <Section title="Every block of the reference models" badges={<Evidence kind="verified" />}>
        <Select label="Model" value={real} onChange={setReal} options={CORE_PROFILES.map((x) => ({ value: x.id, label: x.name }))} />
        <BlockTable p={R} />
      </Section>
    </>
  );
}

function BlockTable({ p }: { p: CoreProfile }) {
  const cat = useCatalog(p.id);
  const a = accounting(p);
  return (
    <div className="table-wrap" style={{ maxHeight: 420, overflowY: 'auto', marginTop: 8 }}>
      <table className="t">
        <thead><tr><th>block</th><th>prefix</th><th className="num">params</th><th>checkpoint shard(s)</th><th>open</th></tr></thead>
        <tbody>
          {Array.from({ length: p.layers }, (_, b) => (
            <tr key={b}>
              <td>{b}</td><td className="mono">model.layers.{b}.*</td><td className="num">{fmtInt(a.perBlock)}</td>
              <td className="mono" style={{ fontSize: '0.8rem' }}>{cat.status === 'ok' ? (cat.data.layerShards[String(b)] ?? []).map((n) => `${String(n).padStart(5, '0')}`).join(', ') + (p.kind === 'moe' && cat.data.expertShardPlacement ? ` · experts in ${[...new Set([...cat.data.expertShardPlacement[String(b)].gate, ...cat.data.expertShardPlacement[String(b)].down])].join(', ')}` : '') : cat.status === 'error' ? 'catalog unavailable' : '…'}</td>
              <td><a href={explorerHref({ m: p.id, lv: 4, b })}>block {b}</a></td>
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
    ...Array.from({ length: p.layers }, (_, l) => ({ kind: 'prefill' as const, layer: l, label: `Prefill · block ${l} (positions 0–${prompt.length - 1} together)` })),
    ...run.decodes.map((d, i) => ({ kind: 'decode' as const, pos: d.pos, label: `Decode ${i + 1} · “${tok(d.token)}” at position ${d.pos}` })),
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
  return (
    <>
      <Section title="9 · Prefill, decode and the K/V cache" badges={<Evidence kind="simulation" />}>
        <ToyControls />
        <PromptEditor />
        <div className="row" role="toolbar" aria-label="Timeline controls" style={{ margin: '8px 0' }}>
          <button onClick={() => setK(Math.max(0, k - 1))} disabled={k === 0}>◀ Previous</button>
          <button onClick={() => setPlaying(!playing)} aria-pressed={playing}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
          <button onClick={() => setK(Math.min(steps.length - 1, k + 1))} disabled={k >= steps.length - 1}>Next ▶</button>
          <Slider label="speed" value={speed} min={0.5} max={3} step={0.5} onChange={setSpeed} format={(v) => `${v}×`} />
          {reduce && <span className="faint" style={{ fontSize: '0.8rem' }}>Reduced motion: no animation; playback only advances steps.</span>}
        </div>
        <div className="timeline" aria-label="Steps">
          {steps.map((s, i) => <button key={i} className={`step${i === k ? ' cur' : i < k ? ' done' : ''}`} onClick={() => setK(i)} aria-current={i === k ? 'step' : undefined}>{s.label}</button>)}
        </div>
        <div className="grid2" style={{ marginTop: 10 }}>
          <div>
            <h3 aria-live="polite">{cur.label}</h3>
            {cur.kind === 'prefill' ? (
              <p>All {prompt.length} prompt positions pass through block {cur.layer} at once (parallel across positions, causal mask inside attention). Block {cur.layer} must wait for block {cur.layer! - 1 >= 0 ? cur.layer! - 1 : 'the embedding'} to finish: blocks are a sequential dependency. Block {cur.layer} stores K and V for every prompt position.</p>
            ) : (
              <p>Only the new token “{tok(run.decodes[k - p.layers].token)}” is processed, through all {p.layers} blocks. Its query reads the cached K/V of positions 0–{cur.pos! - 1} plus its own; its K/V are appended. RoPE uses its absolute position <strong>{cur.pos}</strong> (= cache length before the step). Earlier K/V are reused, not recomputed. Next token chosen greedily from the toy logits.</p>
            )}
            <p className="faint" style={{ fontSize: '0.86rem' }}>The cache is computed runtime state, not trained parameters. It is cleared whenever the weights, seed, configuration or an earlier token change.</p>
          </div>
          <div>
            <h3>Cache contents</h3>
            <table className="t"><thead><tr><th>block</th><th>K shape [Hkv, T, d_h]</th><th>V shape</th><th>positions</th></tr></thead>
              <tbody>{Array.from({ length: p.layers }, (_, l) => {
                const n = cacheLen(l);
                return <tr key={l}><td>{l}</td><td className="mono">[{p.hkv}, {n}, {p.headDim}]</td><td className="mono">[{p.hkv}, {n}, {p.headDim}]</td><td className="mono">{n ? `0–${n - 1}` : 'empty'}{cur.kind === 'decode' && n ? ` (new: ${n - 1})` : ''}</td></tr>;
              })}</tbody>
            </table>
            <p className="faint" style={{ fontSize: '0.84rem' }}>Toy: {p.hkv} K/V head × {p.headDim} values × 2 (K and V) per position per block. Real: {CORE_PROFILES.map((x) => `${x.name} ${fmtBytes(kvBytesPerToken(x))}/token`).join(' · ')} at BF16.</p>
          </div>
        </div>
      </Section>
      <Section title="Correctness check: cached decoding = full-prefix recomputation" badges={<Evidence kind="simulation" />}>
        <p>Two independent code paths: <code>forwardFull</code> (layer-major over every position with an explicit causal mask) and <code>decodeStep</code> (one token through all blocks, reading the cache). Tolerance {CACHE_TOLERANCE}.</p>
        <table className="t"><thead><tr><th>step</th><th>token</th><th>position</th><th className="num">max |logit difference|</th><th>result</th></tr></thead>
          <tbody>{run.decodes.map((d, i) => <tr key={i}><td>decode {i + 1}</td><td className="mono">{tok(d.token)}</td><td>{d.pos}</td><td className="num">{d.diff.toExponential(2)}</td><td>{d.diff <= CACHE_TOLERANCE ? '✓ equal within tolerance' : '✗ MISMATCH'}</td></tr>)}</tbody>
        </table>
        <p><strong>{allOk ? '✓ All decode steps match.' : '✗ A mismatch was found.'}</strong></p>
      </Section>
      <Section title="Cache invalidation (shared simulator cache)" badges={<Evidence kind="simulation" />}>
        <p className="muted">The shared simulator keeps a real K/V cache keyed by its model state. Edit a prompt token above, change the seed, or edit a weight in the Weight lab, then synchronize: incompatible entries are dropped before recomputation.</p>
        <div className="row">
          <button className="primary" onClick={() => dispatch({ type: 'sync' })}>Synchronize cache with current prompt</button>
          <button onClick={() => dispatch({ type: 'clearCache' })}>Clear cache</button>
        </div>
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>Cached positions</dt><dd>{state.cache.tokens.length} ({state.cache.tokens.map(tok).join(' ') || 'empty'})</dd>
          <dt>Cache state key</dt><dd className="mono" style={{ fontSize: '0.78rem' }}>{state.cache.stateKey === stateKey ? 'matches current model state' : 'STALE: model state changed since it was built'}</dd>
          <dt>Prefix match</dt><dd>{(() => { let i = 0; while (i < state.cache.tokens.length && state.cache.tokens[i] === state.tokens[i]) i++; return i === state.cache.tokens.length ? `cache is a prefix of the prompt (${state.tokens.length - i} position(s) still to decode)` : `prompt differs from cache at position ${i}: positions ${i}+ are invalid`; })()}</dd>
          {syncInfo && <><dt>Last action</dt><dd>{syncInfo}</dd></>}
          {state.lastSync && state.cache.tokens.length === state.tokens.length && state.cache.stateKey === stateKey && (
            <><dt>Check</dt><dd>{(() => {
              const last = decodeStepLogits(p, weights, state.cache);
              const diff = maxAbsDiff(last, storeFull[storeFull.length - 1].logits);
              return `${diff <= CACHE_TOLERANCE ? '✓' : '✗'} last-position logits from the cache path vs full recomputation: max difference ${diff.toExponential(2)}`;
            })()}</dd></>
          )}
        </dl>
        <p className="faint" style={{ fontSize: '0.84rem' }}>Inspection never touches this cache: selecting heads, positions or steps above is read-only.</p>
      </Section>
      <Section title="Newer hybrid models need a different state accounting" badges={<Evidence kind="derived" />}>
        <p>Gated DeltaNet blocks keep a fixed-size recurrent state per sequence instead of a growing K/V cache, and Qwen3.8-Flash-Next's n-gram tables are learned parameters, not state. Its attention blocks still keep per-token keys and values. The ordinary formula above therefore does not apply; see the <a href="#/hybrid/qwen3.8-flash-next">Qwen3.8-Flash-Next page</a> and the <a href="#/labs/memory">memory lab</a>.</p>
        <p className="faint">{CORE_PROFILES.map((x) => `${x.name}: ${fmtB(accounting(x).total)} parameters`).join(' · ')}</p>
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
