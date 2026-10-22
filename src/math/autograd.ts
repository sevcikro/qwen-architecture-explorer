// Minimal scalar reverse-mode automatic differentiation (micrograd-style) for the training toy.
export class Value {
  data: number;
  grad = 0;
  readonly prev: Value[];
  readonly op: string;
  label: string;
  private back: () => void = () => {};

  constructor(data: number, prev: Value[] = [], op = '', label = '') {
    this.data = data;
    this.prev = prev;
    this.op = op;
    this.label = label;
  }

  static of(x: number | Value): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(o: number | Value): Value {
    const b = Value.of(o);
    const out = new Value(this.data + b.data, [this, b], '+');
    out.back = () => {
      this.grad += out.grad;
      b.grad += out.grad;
    };
    return out;
  }

  mul(o: number | Value): Value {
    const b = Value.of(o);
    const out = new Value(this.data * b.data, [this, b], '*');
    out.back = () => {
      this.grad += b.data * out.grad;
      b.grad += this.data * out.grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(o: number | Value): Value {
    return this.add(Value.of(o).neg());
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out.back = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], 'log');
    out.back = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /** x^(-1/2) */
  rsqrt(): Value {
    const r = 1 / Math.sqrt(this.data);
    const out = new Value(r, [this], 'rsqrt');
    out.back = () => {
      this.grad += -0.5 * Math.pow(this.data, -1.5) * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = this.data >= 0 ? 1 / (1 + Math.exp(-this.data)) : Math.exp(this.data) / (1 + Math.exp(this.data));
    const out = new Value(s, [this], 'sigmoid');
    out.back = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  /** SiLU(a) = a * sigmoid(a); derivative sigma(a) * (1 + a * (1 - sigma(a))). */
  silu(): Value {
    const a = this.data;
    const s = a >= 0 ? 1 / (1 + Math.exp(-a)) : Math.exp(a) / (1 + Math.exp(a));
    const out = new Value(a * s, [this], 'silu');
    out.back = () => {
      this.grad += s * (1 + a * (1 - s)) * out.grad;
    };
    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const seen = new Set<Value>();
    const stack: [Value, boolean][] = [[this, false]];
    while (stack.length) {
      const [v, done] = stack.pop()!;
      if (done) {
        topo.push(v);
        continue;
      }
      if (seen.has(v)) continue;
      seen.add(v);
      stack.push([v, true]);
      for (const c of v.prev) if (!seen.has(c)) stack.push([c, false]);
    }
    for (const v of topo) v.grad = 0;
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i].back();
  }
}

export const vsum = (xs: Value[]): Value => xs.reduce((a, b) => a.add(b), new Value(0));
export const vdot = (a: Value[], b: Value[]): Value => vsum(a.map((x, i) => x.mul(b[i])));
