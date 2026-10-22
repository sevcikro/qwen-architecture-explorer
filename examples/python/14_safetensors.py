"""Read a .safetensors checkpoint by hand: 8-byte header length, JSON header, raw bytes.
Then read one small tile of a weight matrix and compare with the safetensors library.

Run:  python 14_safetensors.py PATH/TO/model.safetensors      (needs: numpy; safetensors optional)
"""
import json
import struct
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "model.safetensors"
with open(path, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]                 # little-endian unsigned 64-bit length
    header = json.loads(fh.read(n))
    data_start = 8 + n

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    print("header bytes:", n, " tensors:", len(tensors))
    for name in list(tensors)[:4]:
        print(f"   {name:<48} {tensors[name]['dtype']} {tensors[name]['shape']}")
    total = sum(int(np.prod(v["shape"])) for v in tensors.values())
    print("stored elements:", f"{total:,}")

    name = "model.layers.0.self_attn.q_proj.weight"
    info = tensors[name]
    rows, cols = info["shape"]
    begin = info["data_offsets"][0]
    tile = []
    for r in range(4):                                     # a 4 x 4 tile from the top-left corner
        fh.seek(data_start + begin + (r * cols) * 2)       # BF16 = 2 bytes per number, row-major
        raw = np.frombuffer(fh.read(4 * 2), dtype=np.uint16)
        tile.append((raw.astype(np.uint32) << 16).view(np.float32))   # BF16 -> float32 exactly
    tile = np.array(tile)
    print(f"\n{name}[0:4, 0:4] (shape {rows} x {cols}):")
    print(tile)

try:
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        ref = f.get_slice(name)[0:4, 0:4].float().numpy()
    print("matches the safetensors library:", np.array_equal(ref, tile))
except ImportError:
    pass
