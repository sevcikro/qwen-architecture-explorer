import { useState } from 'react';
import { useToy } from '../state/toyStore';
import { TOY_VOCAB, DEFAULT_SEED } from '../math/toy';
import { SimBanner } from '../components/ui';

/** Explicit, state-changing simulator controls (variant, seed, reset). Inspection controls live elsewhere. */
export function ToyControls({ allowVariant = true }: { allowVariant?: boolean }) {
  const { state, dispatch, edited, profile } = useToy();
  const [seedDraft, setSeedDraft] = useState(String(state.seed));
  const [copied, setCopied] = useState(false);
  return (
    <div className="card tight" style={{ marginBottom: 12 }}>
      <SimBanner>
        <span><strong>{profile.name}</strong>: d={profile.d}, {profile.layers} blocks, {profile.hq} query heads / {profile.hkv} K/V head, head dim {profile.headDim}{profile.moe ? `, ${profile.moe.numExperts} experts top-${profile.moe.topK} (width ${profile.moe.expertWidth})` : `, FFN width ${profile.ffnWidth}`}. Not Qwen's sizes ({profile.moe ? 'Qwen3-30B-A3B has 128 experts, top-8' : 'Qwen3-8B has d=4096'}).</span>
      </SimBanner>
      <div className="row" style={{ marginTop: 8 }}>
        {allowVariant && (
          <div className="row" role="group" aria-label="Simulator variant">
            <button className="small" aria-pressed={state.variant === 'toy-dense'} onClick={() => dispatch({ type: 'variant', v: 'toy-dense' })}>Dense toy</button>
            <button className="small" aria-pressed={state.variant === 'toy-moe'} onClick={() => dispatch({ type: 'variant', v: 'toy-moe' })}>MoE toy</button>
          </div>
        )}
        <label className="field inline">Seed
          <input type="text" inputMode="numeric" value={seedDraft} style={{ width: 120 }} onChange={(e) => setSeedDraft(e.target.value.replace(/[^0-9]/g, ''))} aria-describedby="seed-help" />
        </label>
        <button className="small" disabled={!seedDraft || Number(seedDraft) === state.seed} onClick={() => dispatch({ type: 'seed', seed: Number(seedDraft) })}>Apply seed</button>
        <button className="small" onClick={() => { navigator.clipboard?.writeText(String(state.seed)).then(() => setCopied(true), () => setCopied(false)); }}>{copied ? 'Seed copied' : 'Copy seed'}</button>
        <button className="small" onClick={() => { setSeedDraft(String(DEFAULT_SEED)); dispatch({ type: 'resetAll' }); }}>Reset simulator</button>
        {edited && <span className="banner-edit">Edited weights ({state.edits.length}) <button className="small" onClick={() => dispatch({ type: 'resetWeights' })}>restore fixture</button></span>}
      </div>
      <div id="seed-help" className="faint" style={{ fontSize: '0.78rem', marginTop: 4 }}>The same seed always regenerates the same weights. Applying a seed, switching variant or editing a weight changes the model state and invalidates its cache.</div>
    </div>
  );
}

export function PromptEditor({ max = 12 }: { max?: number }) {
  const { state, dispatch } = useToy();
  const toks = state.tokens;
  return (
    <div>
      <div className="faint" style={{ fontSize: '0.85rem' }}>Toy prompt (toy word-level tokenizer, <strong>not</strong> Qwen tokenization):</div>
      <div className="row" style={{ gap: 6, margin: '4px 0' }}>
        {toks.map((t, i) => (
          <label key={i} className="chip" style={{ padding: '0 4px' }}>
            <span className="faint">{i}</span>
            <select aria-label={`token at position ${i}`} value={t} onChange={(e) => dispatch({ type: 'tokens', tokens: toks.map((o, k) => (k === i ? Number(e.target.value) : o)) })} style={{ minHeight: 28, padding: '2px 4px', border: 'none', background: 'transparent' }}>
              {TOY_VOCAB.map((w, id) => <option key={id} value={id}>{w}</option>)}
            </select>
          </label>
        ))}
        <button className="small" disabled={toks.length >= max} onClick={() => dispatch({ type: 'tokens', tokens: [...toks, 11] })}>+ token</button>
        <button className="small" disabled={toks.length <= 1} onClick={() => dispatch({ type: 'tokens', tokens: toks.slice(0, -1) })}>− last</button>
      </div>
    </div>
  );
}

export const tok = (id: number) => TOY_VOCAB[id] ?? `#${id}`;
