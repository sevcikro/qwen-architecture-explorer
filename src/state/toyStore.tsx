import { createContext, useContext, useMemo, useReducer, type ReactNode } from 'react';
import { TOY_DENSE, TOY_MOE, type CoreProfile } from '../models/profiles';
import { DEFAULT_PROMPT, DEFAULT_SEED, emptyCache, forwardFull, initToyWeights, modelStateKey, syncCache, toyTensor, withEditedWeight, type CacheSync, type KvCache, type PositionTrace, type ToyWeights } from '../math/toy';
import { samplerRng } from '../math/sampling';
import type { Rng } from '../math/rng';

// ---------------------------------------------------------------------------
// One shared toy-simulator state for all labs. State-changing actions are explicit;
// inspection (selecting heads, rows, tokens to look at) never touches this store.
// ---------------------------------------------------------------------------

export interface WeightEdit { name: string; row: number; col: number | null; from: number; to: number }

interface ToyState {
  variant: 'toy-dense' | 'toy-moe';
  seed: number;
  edits: WeightEdit[];
  tokens: number[];
  cache: KvCache;
  lastSync: Omit<CacheSync, 'cache' | 'traces'> | null;
  samplerSeed: number;
  rng: Rng;
  draws: number;
  log: string[];
}

type Action =
  | { type: 'variant'; v: ToyState['variant'] }
  | { type: 'seed'; seed: number }
  | { type: 'edit'; name: string; row: number; col: number | null; value: number }
  | { type: 'resetWeights' }
  | { type: 'tokens'; tokens: number[] }
  | { type: 'sync' }
  | { type: 'clearCache' }
  | { type: 'samplerSeed'; seed: number }
  | { type: 'advanceRng'; rng: Rng }
  | { type: 'resetAll' };

const profileOf = (v: ToyState['variant']): CoreProfile => (v === 'toy-dense' ? TOY_DENSE : TOY_MOE);

function weightsFor(variant: ToyState['variant'], seed: number, edits: WeightEdit[]): ToyWeights {
  let w = initToyWeights(profileOf(variant), seed);
  for (const e of edits) w = withEditedWeight(w, e.name, e.row, e.col, e.to);
  return w;
}

const keyOf = (s: Pick<ToyState, 'variant' | 'seed' | 'edits'>) => modelStateKey(profileOf(s.variant), s.seed, s.edits.length) + `|${s.edits.map((e) => `${e.name}:${e.row}:${e.col}=${e.to}`).join(',')}`;

function initial(): ToyState {
  const base = { variant: 'toy-dense' as const, seed: DEFAULT_SEED, edits: [] as WeightEdit[] };
  return { ...base, tokens: DEFAULT_PROMPT.slice(), cache: emptyCache(TOY_DENSE, keyOf(base)), lastSync: null, samplerSeed: 7, rng: samplerRng(7), draws: 0, log: [] };
}

function reducer(s: ToyState, a: Action): ToyState {
  const stamp = (msg: string, st: ToyState): ToyState => ({ ...st, log: [msg, ...st.log].slice(0, 30) });
  switch (a.type) {
    case 'variant':
      if (a.v === s.variant) return s;
      return stamp(`Switched to ${profileOf(a.v).name}; weights, cache and traces of the previous variant discarded.`, { ...s, variant: a.v, edits: [], cache: emptyCache(profileOf(a.v), keyOf({ ...s, variant: a.v, edits: [] })), lastSync: null });
    case 'seed':
      return stamp(`Seed set to ${a.seed}; all weights regenerated; cache invalidated.`, { ...s, seed: a.seed >>> 0, edits: [], lastSync: null });
    case 'edit': {
      const w = weightsFor(s.variant, s.seed, s.edits);
      const t = toyTensor(w, a.name);
      const from = a.col === null ? (t as number[])[a.row] : (t as number[][])[a.row][a.col];
      return stamp(`Edited ${a.name}[${a.row}${a.col === null ? '' : `, ${a.col}`}] ${from.toFixed(4)} → ${a.value}; model state changed; cache invalidated.`, { ...s, edits: [...s.edits, { name: a.name, row: a.row, col: a.col, from, to: a.value }] });
    }
    case 'resetWeights':
      return stamp('Weights reset to the exact seeded fixture.', { ...s, edits: [] });
    case 'tokens':
      return { ...s, tokens: a.tokens.slice(0, profileOf(s.variant).context.maxPositionEmbeddings) };
    case 'sync': {
      const p = profileOf(s.variant);
      const r = syncCache(p, weightsFor(s.variant, s.seed, s.edits), s.cache, keyOf(s), s.tokens);
      const { cache, traces: _t, ...meta } = r;
      void _t;
      const msg = r.reason === 'model-state-changed' ? `Cache rebuilt: model state changed (${r.invalidatedPositions} stale positions dropped).` : r.reason === 'prefix-edited' ? `Prefix edited: kept ${r.reusedPositions} cached positions, dropped ${r.invalidatedPositions}.` : r.reason === 'extended' ? `Decoded ${tokensAdded(r)} new position(s) reusing ${r.reusedPositions} cached.` : 'Cache already consistent; nothing recomputed.';
      return stamp(msg, { ...s, cache, lastSync: meta });
    }
    case 'clearCache':
      return stamp('Cache cleared by user.', { ...s, cache: emptyCache(profileOf(s.variant), keyOf(s)), lastSync: null });
    case 'samplerSeed':
      return { ...s, samplerSeed: a.seed, rng: samplerRng(a.seed), draws: 0 };
    case 'advanceRng':
      return { ...s, rng: a.rng, draws: s.draws + 1 };
    case 'resetAll':
      return initial();
  }
}
const tokensAdded = (r: CacheSync) => r.traces.length;

interface ToyApi {
  state: ToyState;
  profile: CoreProfile;
  weights: ToyWeights;
  stateKey: string;
  edited: boolean;
  trace: PositionTrace[]; // full-prefix recomputation for current tokens (inspection uses this)
  dispatch: (a: Action) => void;
}

const Ctx = createContext<ToyApi | null>(null);

export function ToyProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initial);
  const profile = profileOf(state.variant);
  const weights = useMemo(() => weightsFor(state.variant, state.seed, state.edits), [state.variant, state.seed, state.edits]);
  const stateKey = keyOf(state);
  const trace = useMemo(() => forwardFull(profile, weights, state.tokens), [profile, weights, state.tokens]);
  const api = useMemo<ToyApi>(() => ({ state, profile, weights, stateKey, edited: state.edits.length > 0, trace, dispatch }), [state, profile, weights, stateKey, trace]);
  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useToy(): ToyApi {
  const c = useContext(Ctx);
  if (!c) throw new Error('ToyProvider missing');
  return c;
}

export const toyProfile = profileOf;
