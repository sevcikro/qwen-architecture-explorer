"""Backpropagation by hand for a tiny network, checked against PyTorch autograd
and against a finite-difference estimate.

Run:  python 10_backprop.py      (needs: numpy, torch)
"""
import numpy as np
import torch

# Network: z = w*x + b ; y = SiLU(z) ; loss = (y - t)^2
x, t = 1.5, 0.8
w, b = 0.4, -0.1

sig = lambda v: 1 / (1 + np.exp(-v))
# forward pass, keeping every intermediate value
z = w * x + b
y = z * sig(z)
loss = (y - t) ** 2
print(f"forward : z = {z:.6f}, y = {y:.6f}, loss = {loss:.6f}")

# backward pass: chain rule from the loss back to each weight
dL_dy = 2 * (y - t)
dy_dz = sig(z) * (1 + z * (1 - sig(z)))        # derivative of SiLU
dL_dz = dL_dy * dy_dz
dL_dw = dL_dz * x                               # because dz/dw = x
dL_db = dL_dz * 1.0                             # because dz/db = 1
print(f"by hand : dL/dw = {dL_dw:.6f}, dL/db = {dL_db:.6f}")

# the same with PyTorch's automatic differentiation
wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
lt = (torch.nn.functional.silu(wt * x + bt) - t) ** 2
lt.backward()
print(f"autograd: dL/dw = {wt.grad.item():.6f}, dL/db = {bt.grad.item():.6f}")

# finite differences: nudge w a tiny bit and measure how the loss changes
def L(wv, bv):
    zz = wv * x + bv
    return (zz * sig(zz) - t) ** 2
h = 1e-6
print(f"numeric : dL/dw = {(L(w + h, b) - L(w - h, b)) / (2 * h):.6f}")

# one gradient-descent step: w_new = w - lr * dL/dw (the same for b)
print()
for lr in (0.1, 0.5):
    after = L(w - lr * dL_dw, b - lr * dL_db)
    verdict = "lower" if after < L(w, b) else "HIGHER: the step overshot"
    print(f"lr = {lr}: loss {L(w, b):.6f} -> {after:.6f}  ({verdict})")
