// BF16 = the top 16 bits of an IEEE-754 float32: 1 sign bit, 8 exponent bits, 7 fraction bits.
export interface Bf16Parts {
  bits: number; // 16-bit pattern
  sign: number;
  exponentBits: number;
  fractionBits: number;
  unbiasedExponent: number | null; // null for zero/subnormal/inf/nan classes
  kind: 'zero' | 'subnormal' | 'normal' | 'inf' | 'nan';
  value: number;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** Round-to-nearest-even float -> bf16 bit pattern (what a converter does). */
export function toBf16Bits(x: number): number {
  f32[0] = x;
  const u = u32[0];
  if ((u & 0x7f800000) === 0x7f800000 && (u & 0x007fffff) !== 0) return 0x7fc0; // NaN
  const lsb = (u >>> 16) & 1;
  return ((u + 0x7fff + lsb) >>> 16) & 0xffff;
}

export function fromBf16Bits(bits: number): number {
  u32[0] = (bits & 0xffff) << 16;
  return f32[0];
}

export function bf16Parts(bits: number): Bf16Parts {
  const sign = (bits >>> 15) & 1;
  const exponentBits = (bits >>> 7) & 0xff;
  const fractionBits = bits & 0x7f;
  let kind: Bf16Parts['kind'];
  if (exponentBits === 0) kind = fractionBits === 0 ? 'zero' : 'subnormal';
  else if (exponentBits === 0xff) kind = fractionBits === 0 ? 'inf' : 'nan';
  else kind = 'normal';
  return { bits, sign, exponentBits, fractionBits, unbiasedExponent: kind === 'normal' ? exponentBits - 127 : null, kind, value: fromBf16Bits(bits) };
}

/** True when a float64 value is exactly representable in bf16 (e.g. a value widened from a real bf16 checkpoint). */
export const isExactBf16 = (x: number) => fromBf16Bits(toBf16Bits(x)) === x;

/** Symmetric int-k quantization with one scale per group (illustrative). */
export function quantizeSymmetric(values: number[], bits: number) {
  const qmax = 2 ** (bits - 1) - 1;
  const maxAbs = Math.max(...values.map(Math.abs), 1e-30);
  const scale = maxAbs / qmax;
  const q = values.map((v) => Math.max(-qmax - 1, Math.min(qmax, Math.round(v / scale))));
  return { scale, q, dequant: q.map((x) => x * scale), qmax };
}
