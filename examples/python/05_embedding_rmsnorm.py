"""Embedding lookup (= one-hot times a matrix) and RMSNorm.

Run:  python 05_embedding_rmsnorm.py      (needs: numpy)
"""
import numpy as np

rng = np.random.default_rng(1)
vocab_size, d = 10, 4
E = rng.normal(size=(vocab_size, d))            # embedding table: one row per token id

token_id = 7
row = E[token_id]                               # what a real implementation does: index a row
one_hot = np.zeros(vocab_size); one_hot[token_id] = 1.0
print("lookup     :", np.round(row, 4))
print("one-hot @ E:", np.round(one_hot @ E, 4), "(identical, but wasteful)")

# RMSNorm (as in Qwen3): divide by the root-mean-square, then multiply by a learned gain.
def rmsnorm(x, gain, eps=1e-6):
    rms = np.sqrt(np.mean(x * x) + eps)
    return x / rms * gain

x = np.array([12.0, -3.0, 0.5, 40.0])
gain = np.ones(d)                               # learned; starts at 1
y = rmsnorm(x, gain)
print("\nx          :", x, " rms =", round(float(np.sqrt(np.mean(x * x))), 4))
print("RMSNorm(x) :", np.round(y, 4), " rms =", round(float(np.sqrt(np.mean(y * y))), 4))
print("scale-free :", np.allclose(rmsnorm(10 * x, gain), y, atol=1e-6), "(multiplying x by 10 gives the same output)")
