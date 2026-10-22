"""The SwiGLU feed-forward network and a mixture-of-experts (MoE) layer.

Run:  python 08_ffn_moe.py      (needs: numpy)
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(4)
d, f = 8, 16

def silu(v):
    return v / (1 + np.exp(-v))

def swiglu(x, Wg, Wu, Wd):
    """Qwen3 MLP: down( SiLU(gate(x)) * up(x) ), no biases."""
    return Wd @ (silu(Wg @ x) * (Wu @ x))

x = rng.normal(size=d)
Wg, Wu = rng.normal(size=(f, d)) * 0.3, rng.normal(size=(f, d)) * 0.3
Wd = rng.normal(size=(d, f)) * 0.3
print("dense FFN output:", swiglu(x, Wg, Wu, Wd))
print("dense FFN parameters:", Wg.size + Wu.size + Wd.size)

# ---------------- MoE: a router picks top-k experts per token ----------------
E, k, fe = 8, 2, 8                       # 8 small experts, 2 used per token
experts = [(rng.normal(size=(fe, d)) * 0.3, rng.normal(size=(fe, d)) * 0.3, rng.normal(size=(d, fe)) * 0.3) for _ in range(E)]
W_router = rng.normal(size=(E, d))

logits = W_router @ x
p = np.exp(logits - logits.max()); p /= p.sum()      # softmax over all experts
top = np.argsort(-p)[:k]                             # indices of the k largest
alpha = p[top] / p[top].sum()                        # renormalize (norm_topk_prob = True in Qwen3-MoE)
y = sum(a * swiglu(x, *experts[e]) for a, e in zip(alpha, top))

print("\nrouter probabilities:", p)
print("chosen experts:", top, " mixture weights:", alpha, " sum:", alpha.sum())
print("MoE output:", y)
stored = sum(m.size for ex in experts for m in ex) + W_router.size
used = sum(m.size for e in top for m in experts[e]) + W_router.size
print(f"parameters stored: {stored}, used for this token: {used}")
