"""Logits -> probabilities (softmax), temperature, and the training loss (cross-entropy).

Run:  python 03_softmax_xent.py      (needs: numpy)
"""
import numpy as np

vocab = ["Paris", "London", "Rome", "banana"]
logits = np.array([4.0, 2.5, 2.0, -1.0])       # raw scores from the output head

def softmax(z):
    z = z - z.max()                  # subtracting the max avoids overflow; result is unchanged
    e = np.exp(z)
    return e / e.sum()

p = softmax(logits)
for w, pi in zip(vocab, p):
    print(f"{w:<7} p = {pi:.4f}")
print("sum =", round(float(p.sum()), 6))

print("\ntemperature changes how peaked the distribution is:")
for T in (0.5, 1.0, 2.0):
    print(f"  T = {T}:", np.round(softmax(logits / T), 4))

# Cross-entropy: the loss for one position is -log p(correct next token).
target = vocab.index("Paris")
print("\nloss if the answer is 'Paris'  =", round(float(-np.log(p[target])), 4))
print("loss if the answer is 'banana' =", round(float(-np.log(p[vocab.index('banana')])), 4))
print("a model that guesses uniformly over 151,936 tokens has loss ln(151936) =", round(float(np.log(151936)), 3))
