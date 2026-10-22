import { TOY_DENSE, TOY_MOE, type CoreProfile } from '../models/profiles';
import { tensorFamilies, tensorName, type TensorFamily } from '../models/shapes';
import { forwardFull, initToyWeights, toyTensor, type PositionTrace, type ToyWeights } from '../math/toy';
import { swiglu } from '../math/ops';
import type { Vec } from '../math/linalg';

export interface Counterpart {
  toy: CoreProfile;
  family: TensorFamily;
  layer: number;
  expert: number;
  row: number;
  col: number | null;
  name: string;
  mapped: boolean; // true when indices were folded into the toy's smaller extent
}

/** Map a selection in any profile to the same tensor role in the matching toy simulator. */
export function toyCounterpart(p: CoreProfile, key: string, b = 0, e = 0, r = 0, c: number | null = 0): Counterpart | null {
  const toy = p.status === 'toy' ? p : p.kind === 'dense' ? TOY_DENSE : TOY_MOE;
  const fam = tensorFamilies(toy).find((f) => f.key === key);
  if (!fam) return null;
  const layer = Math.min(b, toy.layers - 1);
  const expert = toy.moe ? e % toy.moe.numExperts : 0;
  const row = r % fam.shape[0];
  const col = fam.shape.length > 1 ? (c ?? 0) % fam.shape[1] : null;
  const mapped = p.status !== 'toy' || layer !== b || row !== r || (col ?? 0) !== (c ?? 0) || (toy.moe ? expert !== e : false);
  return { toy, family: fam, layer, expert, row, col, name: tensorName(fam, layer, expert), mapped };
}

/** The input vector that the selected weight row multiplies, taken from a toy forward pass. */
export function unitInput(cp: Counterpart, trace: PositionTrace[], w: ToyWeights, pos: number): { x: Vec; label: string } | null {
  const P = trace[Math.min(pos, trace.length - 1)];
  const L = P.layers[cp.layer];
  switch (cp.family.key) {
    case 'q_proj': case 'k_proj': case 'v_proj': return { x: L.ln1, label: `RMSNorm(x) at position ${P.pos}, block ${cp.layer}` };
    case 'o_proj': return { x: L.attn.concat, label: `concatenated head outputs at position ${P.pos}, block ${cp.layer}` };
    case 'gate_proj': case 'up_proj': case 'router': case 'expert_gate_proj': case 'expert_up_proj': return { x: L.ln2, label: `RMSNorm(h) at position ${P.pos}, block ${cp.layer}` };
    case 'down_proj': return { x: L.ffn!.z, label: `gated channels z at position ${P.pos}, block ${cp.layer}` };
    case 'expert_down_proj': {
      const ex = w.layers[cp.layer].experts![cp.expert];
      return { x: swiglu(L.ln2, ex.gate, ex.up, ex.down).z, label: `expert ${cp.expert}'s gated channels at position ${P.pos} (computed even if the router did not select it, for illustration)` };
    }
    case 'lm_head': return { x: P.finalNorm, label: `final RMSNorm output at position ${P.pos}` };
    default: return null;
  }
}

export function toyTrace(toy: CoreProfile, seed: number, tokens: number[], shared?: { variant: string; weights: ToyWeights }) {
  const w = shared && shared.variant === toy.id ? shared.weights : initToyWeights(toy, seed);
  return { w, trace: forwardFull(toy, w, tokens), shared: !!shared && shared.variant === toy.id };
}

export function toyValue(w: ToyWeights, cp: Counterpart): number {
  const t = toyTensor(w, cp.name);
  return cp.col === null ? (t as number[])[cp.row] : (t as number[][])[cp.row][cp.col];
}
