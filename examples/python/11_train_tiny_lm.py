"""Train a tiny character-level transformer language model with PyTorch.
It learns to continue one short text: a toy, but the loop is the real recipe
(next-token targets, cross-entropy, backpropagation, AdamW updates).

Run:  python 11_train_tiny_lm.py      (needs: torch; runs on a CPU in well under a minute)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
text = "the cat sat on the mat. the dog sat on the log. " * 20
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
V, T, d = len(chars), 32, 64

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)            # learned positions (Qwen uses RoPE instead)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=4 * d, batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        n = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(n))
        causal = nn.Transformer.generate_square_subsequent_mask(n)
        return self.head(self.norm(self.blocks(x, mask=causal, is_causal=True)))

model = TinyLM()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print("parameters:", sum(p.numel() for p in model.parameters()))

for step in range(301):
    ix = torch.randint(0, len(data) - T - 1, (16,))
    x = torch.stack([data[i : i + T] for i in ix])          # inputs
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])  # targets = inputs shifted by one
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()                                         # gradients for every weight
    opt.step()                                              # AdamW update
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}")

model.eval()
idx = torch.tensor([[stoi[c] for c in "the d"]])
with torch.no_grad():
    for _ in range(25):
        nxt = model(idx[:, -T:])[0, -1].argmax()            # greedy decoding
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
print("sample:", "".join(chars[i] for i in idx[0].tolist()))
