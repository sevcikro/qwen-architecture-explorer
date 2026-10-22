#!/usr/bin/env python3
"""Compact per-tensor byte locations from extracted safetensors headers.

Input : headers-<model>.json produced by extract_checkpoint_metadata.py
Output: offsets-<model>.json  {"shards": [...], "headerBytes": [...], "t": {name: [shardIdx, dataStart, dtype, shape]}}
Byte position of element (r, c) of a 2-D BF16 tensor inside its shard file:
    8 + headerBytes[shardIdx] + dataStart + (r * cols + c) * 2
(This is a FILE offset. It is not a tensor coordinate and not a GPU memory address.)
Usage: python -I scripts/compact_offsets.py IN_DIR OUT_DIR model-key [model-key...]
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for key in sys.argv[3:]:
    h = json.loads((src / f"headers-{key}.json").read_text())
    shards = sorted({v["shard"] for v in h.values()})
    hb = [next(v["headerBytes"] for v in h.values() if v["shard"] == s) for s in shards]
    t = {n: [shards.index(v["shard"]), v["offsets"][0], v["dtype"], v["shape"]] for n, v in sorted(h.items())}
    (dst / f"offsets-{key}.json").write_text(json.dumps({"shards": shards, "headerBytes": hb, "t": t}, separators=(",", ":")))
    print(key, len(t), (dst / f"offsets-{key}.json").stat().st_size)
