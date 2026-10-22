"""Weight quantization: store numbers with fewer bits plus a scale per group,
and measure the error it introduces.

Run:  python 15_quantize.py      (needs: numpy)
"""
import numpy as np

rng = np.random.default_rng(6)
w = rng.normal(scale=0.02, size=(256, 256)).astype(np.float32)   # weight-like numbers

def quantize(x, bits, group=64):
    """Symmetric per-group quantization: q = round(x / scale), scale = max|x| / (2^(bits-1) - 1)."""
    flat = x.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(flat / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale.astype(np.float16)

def dequantize(q, scale, shape):
    return (q.astype(np.float32) * scale.astype(np.float32)).reshape(shape)

x = rng.normal(size=256).astype(np.float32)                        # an activation vector
y_ref = w @ x
for bits in (8, 4, 3):
    q, s = quantize(w, bits)
    w_hat = dequantize(q, s, w.shape)
    payload = w.size * bits / 8 + s.size * 2                        # bytes: integers + fp16 scales
    rel = np.linalg.norm(w_hat @ x - y_ref) / np.linalg.norm(y_ref)
    print(f"{bits}-bit: {payload / 1024:6.1f} KiB (BF16 would be {w.size * 2 / 1024:.1f} KiB), "
          f"mean |weight error| = {np.abs(w_hat - w).mean():.2e}, relative output error = {rel:.2%}")
print("first weights:", w[0, :4], "\n4-bit version:", dequantize(*quantize(w, 4), w.shape)[0, :4])
