import { describe, expect, it } from 'vitest';
import {
  PARAM_SHAPES, TRAIN_SEQUENCE, finiteDifference, getParam, initTrainParams, lossAndGrad, sgdStep, trainForward, type ParamName, type ParamRef,
} from '../../src/math/train';
import { gatedDeltaRule, qsaSelect, toyGdnInputs, gdnGate } from '../../src/math/hybridToy';
import { sigmoid } from '../../src/math/ops';

describe('training toy', () => {
  const p = initTrainParams(11);

  it('autograd loss equals the independent plain-number forward', () => {
    expect(lossAndGrad(p, TRAIN_SEQUENCE).loss).toBeCloseTo(trainForward(p, TRAIN_SEQUENCE).loss, 12);
  });

  it('teacher forcing: targets are the inputs shifted by one', () => {
    const f = trainForward(p, TRAIN_SEQUENCE);
    expect(f.positions.length).toBe(TRAIN_SEQUENCE.length - 1);
    f.positions.forEach((x, t) => expect(x.target).toBe(TRAIN_SEQUENCE[t + 1]));
  });

  it('every gradient tensor passes a central finite-difference check', () => {
    const { grad } = lossAndGrad(p, TRAIN_SEQUENCE);
    for (const tensor of Object.keys(PARAM_SHAPES) as ParamName[]) {
      const sh = PARAM_SHAPES[tensor];
      const refs: ParamRef[] = sh.length === 1 ? [0, 1, sh[0] - 1].map((row) => ({ tensor, row, col: null })) : [[0, 0], [1, 2], [sh[0] - 1, sh[1] - 1]].map(([row, col]) => ({ tensor, row, col }));
      for (const r of refs) {
        const g = getParam(grad, r);
        const fd = finiteDifference(p, TRAIN_SEQUENCE, r);
        expect(Math.abs(g - fd), `${tensor}[${r.row},${r.col}] ag=${g} fd=${fd}`).toBeLessThan(1e-6 + 1e-4 * Math.abs(fd));
      }
    }
  });

  it('an SGD step changes weights by exactly -lr * grad and the actual new loss is recomputed', () => {
    const { grad, loss } = lossAndGrad(p, TRAIN_SEQUENCE);
    const p2 = sgdStep(p, grad, 0.1);
    const r: ParamRef = { tensor: 'gate', row: 2, col: 1 };
    expect(getParam(p2, r)).toBeCloseTo(getParam(p, r) - 0.1 * getParam(grad, r), 14);
    const small = trainForward(p2, TRAIN_SEQUENCE).loss;
    expect(small).toBeLessThan(loss); // small step on this fixture reduces loss (measured, not assumed in the UI)
    const huge = trainForward(sgdStep(p, grad, 200), TRAIN_SEQUENCE).loss;
    expect(Number.isFinite(huge)).toBe(true);
    expect(initTrainParams(11)).toEqual(p); // reset restores the exact seeded fixture
  });
});

describe('hybrid toy analogues', () => {
  it('gated delta rule: zero decay-free state with beta=1 recalls the written value exactly', () => {
    const k = [[1, 0, 0, 0]], v = [[2, -1, 0.5, 3]];
    const steps = gatedDeltaRule(k, k, v, [0], [1]);
    // q = k (L2-normalized) scaled by 1/sqrt(dk)=0.5 -> output is 0.5 * v
    steps[0].out.forEach((o, j) => expect(o).toBeCloseTo(0.5 * v[0][j], 5));
  });
  it('gate g = -exp(A_log)*softplus(a+dt_bias) is negative so decay exp(g) lies in (0,1)', () => {
    const g = gdnGate(Math.log(4), 0.3, 1);
    expect(g).toBeLessThan(0);
    expect(Math.exp(g)).toBeGreaterThan(0);
    expect(Math.exp(g)).toBeLessThan(1);
    const inp = toyGdnInputs(3, 6);
    const steps = gatedDeltaRule(inp.q, inp.k, inp.v, inp.a.map((a) => gdnGate(0, a, 0)), inp.b.map(sigmoid));
    expect(steps.length).toBe(6);
    expect(steps[5].stateAfter.length).toBe(4);
  });
  it('QSA selection keeps at most budget/ratio complete blocks and always the tail', () => {
    const keys = Array.from({ length: 23 }, (_, i) => [Math.sin(i), Math.cos(i * 0.7), (i % 5) - 2, 0.1 * i]);
    const sel = qsaSelect([[1, 0, 0.5, 0], [0, 1, 0, 0.2]], keys, 23, 8, 4);
    expect(sel.blocks.length).toBe(5);
    expect(sel.blocks.filter((b) => b.selected).length).toBe(2);
    expect(sel.tail).toEqual([20, 21, 22]);
    expect(sel.selectedTokens.length).toBe(2 * 4 + 3);
  });
});
