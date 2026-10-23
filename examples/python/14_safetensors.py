"""Read a .safetensors checkpoint by hand: 8-byte header length, JSON header, raw bytes.
Then read one small tile of a weight matrix and compare with the safetensors library.

Run:  python 14_safetensors.py PATH/TO/model.safetensors      (needs: numpy; safetensors optional)
Without a path it uses $QWEN_MODEL_DIR/model.safetensors, or downloads Qwen3-0.6B's file (about 1.5 GB)
from the Hugging Face Hub at a pinned revision (needs: huggingface_hub).
"""
import json
import os
import struct
import sys
import numpy as np

args = [a for a in sys.argv[1:] if a.endswith(".safetensors")]   # (Jupyter passes its own "-f ..." arguments)
if args:
    path = args[0]
elif os.environ.get("QWEN_MODEL_DIR"):
    path = os.path.join(os.environ["QWEN_MODEL_DIR"], "model.safetensors")
else:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors", revision="c1899de289a04d12100db370d81485cdf75e47ca")
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
