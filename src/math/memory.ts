import type { CoreProfile } from '../models/profiles';
import { accounting } from '../models/shapes';

export const GB = 1e9;
export const GiB = 1024 ** 3;
export const KiB = 1024;

/** Raw theoretical payload: parameters * bits / 8. Not a file size. */
export function weightBytes(params: number, bitsPerParam: number): number {
  return (params * bitsPerParam) / 8;
}

export interface QuantSpec {
  bits: number; // bits per quantized weight
  groupSize: number; // weights sharing one scale
  scaleBits: number; // storage of each scale
  zeroPointBits: number; // 0 for symmetric schemes
  keepUnquantized: { embedding: boolean; head: boolean; norms: boolean };
  unquantizedBits: number;
}

export interface QuantEstimate {
  quantizedParams: number;
  unquantizedParams: number;
  payloadBytes: number;
  metadataBytes: number;
  unquantizedBytes: number;
  totalBytes: number;
}

/** Ideal quantized storage, with metadata and unquantized tensors kept separate. Still not a guaranteed file size. */
export function quantizedEstimate(p: CoreProfile, q: QuantSpec): QuantEstimate {
  const a = accounting(p);
  const norms = a.normsPerBlock * p.layers + a.finalNorm;
  let unq = 0;
  if (q.keepUnquantized.embedding) unq += a.embedding;
  if (q.keepUnquantized.head) unq += a.head;
  if (q.keepUnquantized.norms) unq += norms;
  const quantizedParams = a.total - unq;
  const payloadBytes = weightBytes(quantizedParams, q.bits);
  const groups = Math.ceil(quantizedParams / q.groupSize);
  const metadataBytes = (groups * (q.scaleBits + q.zeroPointBits)) / 8;
  const unquantizedBytes = weightBytes(unq, q.unquantizedBits);
  return { quantizedParams, unquantizedParams: unq, payloadBytes, metadataBytes, unquantizedBytes, totalBytes: payloadBytes + metadataBytes + unquantizedBytes };
}

/** K/V-only cache estimate for ordinary full-attention Qwen3 blocks. Excludes allocator overhead and workspaces. */
export function kvCacheBytes(p: CoreProfile, opts: { batch: number; tokens: number; bytesPerScalar: number }): number {
  return 2 * opts.batch * p.layers * opts.tokens * p.hkv * p.headDim * opts.bytesPerScalar;
}

export function kvBytesPerToken(p: CoreProfile, bytesPerScalar = 2): number {
  return kvCacheBytes(p, { batch: 1, tokens: 1, bytesPerScalar });
}

export interface TrainingMemory {
  weights: number;
  gradients: number;
  optimizerStates: number;
  masterWeights: number;
  total: number;
}

/** Persistent training memory for weights + grads + Adam moments (+ optional fp32 master copy). Activations are separate and workload-dependent. */
export function trainingMemory(params: number, o: { weightBytes: number; gradBytes: number; adamStateBytes: number; masterCopyBytes: number }): TrainingMemory {
  const weights = params * o.weightBytes;
  const gradients = params * o.gradBytes;
  const optimizerStates = params * 2 * o.adamStateBytes;
  const masterWeights = params * o.masterCopyBytes;
  return { weights, gradients, optimizerStates, masterWeights, total: weights + gradients + optimizerStates + masterWeights };
}

export const fmtBytes = (b: number) => {
  const gb = b / GB, gib = b / GiB;
  if (b < 1e6) return `${(b / KiB).toFixed(b < 1e4 ? 2 : 1)} KiB (${b.toLocaleString('en-US')} bytes)`;
  return `${gb.toFixed(gb < 10 ? 3 : 2)} GB = ${gib.toFixed(gib < 10 ? 3 : 2)} GiB`;
};

// ---------------------------------------------------------------------------
// Qwen3.8-Flash-Next per-sequence runtime state (derived from the implementation, see hybrid profile).
// ---------------------------------------------------------------------------
export interface HybridStateSpec {
  qsaLayers: number;
  qsaKvHeads: number;
  qsaHeadDim: number;
  indexerKeyDim: number;
  gdnLayers: number;
  gdnValueHeads: number;
  gdnKeyDim: number;
  gdnValueDim: number;
  gdnConvChannels: number;
  gdnConvKernel: number;
  pleLayers: number;
  pleConvChannels: number;
  pleConvStateLen: number;
  ngramContext: number;
}

export function hybridStateBytes(s: HybridStateSpec, o: { tokens: number; batch: number; kvBytes: number; stateBytes: number }) {
  const kvPerToken = s.qsaLayers * 2 * s.qsaKvHeads * s.qsaHeadDim * o.kvBytes;
  const indexerPerToken = s.qsaLayers * s.indexerKeyDim * o.kvBytes;
  const growing = o.batch * o.tokens * (kvPerToken + indexerPerToken);
  const recurrent = o.batch * s.gdnLayers * s.gdnValueHeads * s.gdnKeyDim * s.gdnValueDim * o.stateBytes;
  const conv = o.batch * (s.gdnLayers * s.gdnConvChannels * (s.gdnConvKernel - 1) + s.pleLayers * s.pleConvChannels * s.pleConvStateLen) * o.kvBytes;
  return { kvPerToken, indexerPerToken, growing, recurrent, conv, total: growing + recurrent + conv };
}
