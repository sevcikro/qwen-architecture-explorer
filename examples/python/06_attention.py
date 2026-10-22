"""Causal self-attention for one head, step by step, with NumPy.

Run:  python 06_attention.py      (needs: numpy)
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(2)
T, d, dh = 4, 8, 4                     # 4 tokens, model width 8, head width 4
X = rng.normal(size=(T, d))            # one row per token (the residual stream after RMSNorm)

Wq, Wk, Wv = (rng.normal(size=(dh, d)) * 0.5 for _ in range(3))
Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T  # [T, dh] each

scores = Q @ K.T / np.sqrt(dh)         # [T, T]: how well query i matches key j
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
scores[mask] = -np.inf                 # causal mask: a token may not look at the future

def softmax_rows(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

A = softmax_rows(scores)               # attention weights, each row sums to 1
print("attention weights (row = query position, column = key position):")
print(A)
print("row sums:", A.sum(axis=1))

out = A @ V                            # each output row is a weighted average of value rows
print("\noutput for the last token = weighted average of V rows:")
print(out[-1])
print("check by hand:", sum(A[-1, j] * V[j] for j in range(T)))
