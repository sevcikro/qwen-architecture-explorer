"""Vectors, dot products and cosine similarity.

Run:  python 02_vectors.py      (needs: numpy)
"""
import numpy as np

# Toy 3-number "meaning vectors" (invented for illustration, not from a model).
words = {
    "cat":   np.array([0.9, 0.8, 0.1]),
    "dog":   np.array([0.8, 0.9, 0.2]),
    "car":   np.array([0.1, 0.2, 0.9]),
}

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

for a in words:
    for b in words:
        if a < b:
            print(f"{a:>3} · {b:<3}  dot = {words[a] @ words[b]:.3f}   cosine = {cosine(words[a], words[b]):.3f}")

# The dot product is "length × length × cos(angle)":
a, b = np.array([3.0, 0.0]), np.array([2.0, 2.0])
angle = np.degrees(np.arccos(cosine(a, b)))
print("\na =", a, " b =", b)
print("a · b               =", a @ b)
print("|a| |b| cos(angle)  =", round(float(np.linalg.norm(a) * np.linalg.norm(b) * np.cos(np.radians(angle))), 6))
print("angle between them  =", round(float(angle), 2), "degrees")
