"""One neuron, then a whole layer of neurons, with NumPy.

Run:  python 01_neuron.py      (needs: numpy)
"""
import numpy as np

# --- one neuron: weighted sum + bias, then an activation -------------------
x = np.array([0.5, -1.0, 2.0])      # three input numbers (activations)
w = np.array([0.8, 0.1, -0.4])      # three learned weights
b = 0.2                             # one learned bias

z = w @ x + b                       # weighted sum: 0.8*0.5 + 0.1*(-1) + (-0.4)*2 + 0.2
print("weighted sum z =", round(float(z), 4))

def silu(v):
    """SiLU (also called swish), the activation used inside Qwen3's FFN."""
    return v / (1.0 + np.exp(-v))

print("SiLU(z)        =", round(float(silu(z)), 4))
print("ReLU(z)        =", round(float(max(z, 0.0)), 4))

# --- a layer = many neurons that read the same input ------------------------
# Each ROW of W holds the weights of one neuron (PyTorch nn.Linear stores
# weight with shape [out_features, in_features] and computes y = x @ W.T).
rng = np.random.default_rng(0)
W = rng.normal(size=(4, 3))         # 4 neurons, 3 inputs each
bias = np.zeros(4)
y = W @ x + bias                    # one matrix-vector product = 4 weighted sums
print("layer output y =", np.round(y, 4))
print("row 0 by hand  =", round(float(W[0] @ x), 4), "(same as y[0])")
print("parameters     =", W.size + bias.size)
