"""Grouped-query attention (GQA) shapes and rotary position embedding (RoPE).

Run:  python 07_gqa_rope.py      (needs: numpy)
"""
import numpy as np

# ---------------- GQA: several query heads share one key/value head ----------------
Hq, Hkv, dh, T = 32, 8, 128, 5          # Qwen3-8B: 32 query heads, 8 K/V heads, head width 128
group = Hq // Hkv
rng = np.random.default_rng(3)
K = rng.normal(size=(Hkv, T, dh))
K_for_queries = np.repeat(K, group, axis=0)    # what "repeat_kv" does in Hugging Face code
print("stored K per layer :", K.shape)
print("K seen by queries  :", K_for_queries.shape, f"(each K/V head serves {group} query heads)")
print("query head 13 uses K/V head", 13 // group)

# ---------------- RoPE: rotate coordinate pairs by a position-dependent angle --------
def rope(x, pos, theta=1_000_000.0):
    """Half-split RoPE as in Qwen3 (rotate_half): pair i with i + d/2."""
    d = x.shape[-1]
    inv_freq = theta ** (-np.arange(0, d, 2) / d)       # d/2 frequencies
    ang = pos * inv_freq
    cos, sin = np.cos(np.concatenate([ang, ang])), np.sin(np.concatenate([ang, ang]))
    x1, x2 = x[: d // 2], x[d // 2 :]
    rotated_half = np.concatenate([-x2, x1])
    return x * cos + rotated_half * sin

q, k = rng.normal(size=dh), rng.normal(size=dh)
print("\nRoPE keeps lengths:", round(float(np.linalg.norm(q)), 6), "->", round(float(np.linalg.norm(rope(q, 7))), 6))
# The score between a query at position m and a key at position n depends only on m - n:
for m, n in [(5, 3), (105, 103), (1005, 1003)]:
    print(f"q at {m:>4}, k at {n:>4} (distance {m - n}):  score = {rope(q, m) @ rope(k, n):.6f}")
print("distance 10 instead:                       score =", round(float(rope(q, 13) @ rope(k, 3)), 6))
