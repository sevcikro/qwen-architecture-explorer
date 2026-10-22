// Small dense helpers. Vectors are number[]; matrices are row-major number[][] with PyTorch [out, in] orientation.
export type Vec = number[];
export type Mat = number[][];

export const zeros = (n: number): Vec => new Array(n).fill(0);
export const dot = (a: Vec, b: Vec): number => {
  if (a.length !== b.length) throw new Error(`dot length mismatch ${a.length} vs ${b.length}`);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
/** y = x W^T for W stored as [out, in]. */
export const linear = (x: Vec, W: Mat): Vec => W.map((row) => dot(x, row));
export const add = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i]);
export const mul = (a: Vec, b: Vec): Vec => a.map((v, i) => v * b[i]);
export const scale = (a: Vec, s: number): Vec => a.map((v) => v * s);
export const maxAbsDiff = (a: Vec, b: Vec): number => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
export const cloneMat = (m: Mat): Mat => m.map((r) => r.slice());
export const shapeOf = (m: Mat): [number, number] => [m.length, m[0]?.length ?? 0];
