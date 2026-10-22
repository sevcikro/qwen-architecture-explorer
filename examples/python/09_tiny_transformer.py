"""A complete (tiny) Qwen3-style decoder in ~80 lines of NumPy, with greedy generation
and a K/V cache. Same operations as Qwen3: embedding, pre-RMSNorm, GQA attention with
QK-norm and RoPE, SwiGLU FFN, residual adds, final norm, untied output head.
Weights are random, so the generated "text" is meaningless; the mechanics are real.

Run:  python 09_tiny_transformer.py      (needs: numpy)
"""
import numpy as np

V, d, L, Hq, Hkv, dh, f = 12, 16, 2, 4, 2, 4, 32       # vocab, width, blocks, heads, head dim, FFN width
rng = np.random.default_rng(5)
W = lambda *s: rng.normal(size=s) * 0.4
params = {
    "embed": W(V, d), "head": W(V, d), "final_norm": np.ones(d),
    "blocks": [dict(n1=np.ones(d), n2=np.ones(d), q=W(Hq * dh, d), k=W(Hkv * dh, d), v=W(Hkv * dh, d),
                    o=W(d, Hq * dh), qn=np.ones(dh), kn=np.ones(dh), g=W(f, d), u=W(f, d), dn=W(d, f)) for _ in range(L)],
}

def rms(x, g, eps=1e-6): return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g
def silu(x): return x / (1 + np.exp(-x))
def rope(x, pos, theta=1e6):                          # x: [..., dh]
    half = x.shape[-1] // 2
    ang = pos * theta ** (-np.arange(half) / half)
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

def step(token, pos, cache):
    """Process ONE token at position `pos`, reading and extending the K/V cache."""
    x = params["embed"][token]
    for l, B in enumerate(params["blocks"]):
        h = rms(x, B["n1"])
        q = rms((B["q"] @ h).reshape(Hq, dh), B["qn"])
        k = rms((B["k"] @ h).reshape(Hkv, dh), B["kn"])
        v = (B["v"] @ h).reshape(Hkv, dh)
        q, k = rope(q, pos), rope(k, pos)
        cache[l]["k"].append(k); cache[l]["v"].append(v)
        Ks, Vs = np.stack(cache[l]["k"], 1), np.stack(cache[l]["v"], 1)   # [Hkv, T, dh]
        heads = []
        for hq in range(Hq):
            kv = hq // (Hq // Hkv)                                         # GQA: shared K/V head
            s = Ks[kv] @ q[hq] / np.sqrt(dh)
            a = np.exp(s - s.max()); a /= a.sum()
            heads.append(a @ Vs[kv])
        x = x + B["o"] @ np.concatenate(heads)                             # residual add 1
        h = rms(x, B["n2"])
        x = x + B["dn"] @ (silu(B["g"] @ h) * (B["u"] @ h))                # residual add 2
    return params["head"] @ rms(x, params["final_norm"])                   # logits [V]

def generate(prompt, n_new):
    cache = [dict(k=[], v=[]) for _ in range(L)]
    for pos, t in enumerate(prompt):                                       # prefill (here token by token)
        logits = step(t, pos, cache)
    out = list(prompt)
    for _ in range(n_new):                                                 # decode: greedy argmax
        nxt = int(np.argmax(logits))
        out.append(nxt)
        logits = step(nxt, len(out) - 1, cache)
    return out, logits

def logits_without_cache(tokens):
    cache = [dict(k=[], v=[]) for _ in range(L)]
    for pos, t in enumerate(tokens):
        logits = step(t, pos, cache)
    return logits

tokens, last = generate([1, 4, 2], n_new=5)
print("generated token ids:", tokens)
print("parameters:", sum(a.size for a in [params["embed"], params["head"], params["final_norm"]]) +
      sum(a.size for B in params["blocks"] for a in B.values()))
print("cache path == full recomputation:", np.allclose(last, logits_without_cache(tokens)))
