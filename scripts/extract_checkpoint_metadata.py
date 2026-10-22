#!/usr/bin/env python3
"""Bounded Safetensors header + sample-tile extraction for Qwen Atlas.

What it does (and does not do):
  * Reads the pinned model.safetensors.index.json of each model.
  * For every shard, reads ONLY the 8-byte length prefix and the JSON header
    using HTTP Range requests. Every response must be HTTP 206 with the exact
    expected length; otherwise that path stops (no silent full-file download).
  * Optionally reads small tiles (default 16 x 16 BF16 values) from named
    tensors, one row-range request per tile row.
  * Enforces a hard aggregate byte budget (default 100 MB) BEFORE accepting
    response bodies, plus per-request timeouts.
  * Never needs or sends credentials: these repositories are public.

Usage (stdlib only, run with -I so nothing is imported from the CWD):
  python -I scripts/extract_checkpoint_metadata.py --out OUT_DIR [--budget-mb 100] [--no-samples]

Outputs:
  OUT_DIR/headers-<model>.json   full per-tensor header table (large; not shipped)
  OUT_DIR/catalog-<model>.json   compact family/shape/dtype/shard catalog (shipped)
  OUT_DIR/samples.json           real tiles with provenance (shipped)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS = {
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"),
    "qwen3.8-flash-next": ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540"),
}

# (model key, tensor name, row0, col0) -- tiles are 16 x 16
SAMPLE_TILES = [
    ("qwen3-8b", "model.layers.0.self_attn.q_proj.weight", 0, 0),
    ("qwen3-8b", "model.layers.0.mlp.gate_proj.weight", 0, 0),
    ("qwen3-8b", "model.embed_tokens.weight", 0, 0),
    ("qwen3-8b", "model.layers.0.input_layernorm.weight", 0, 0),
    ("qwen3-30b-a3b", "model.layers.0.self_attn.q_proj.weight", 0, 0),
    ("qwen3-30b-a3b", "model.layers.0.mlp.gate.weight", 0, 0),
    ("qwen3-30b-a3b", "model.layers.0.mlp.experts.0.gate_proj.weight", 0, 0),
]
TILE = 16
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "BOOL": 1}
UA = "qwen-atlas-extractor/1.0 (+educational; bounded range reads)"
TIMEOUT = 45


class Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def reserve(self, n: int) -> None:
        if n < 0 or self.used + n > self.limit:
            raise RuntimeError(f"byte budget exceeded: used={self.used} request={n} limit={self.limit}")
        self.used += n


def url_for(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def get_small(url: str, budget: Budget, max_bytes: int = 8_000_000) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        declared = int(r.headers.get("Content-Length") or -1)
        if declared < 0 or declared > max_bytes:
            raise RuntimeError(f"refusing {url}: declared length {declared} exceeds {max_bytes}")
        budget.reserve(declared)
        body = r.read(declared + 1)
        if len(body) != declared:
            raise RuntimeError(f"length mismatch for {url}")
        return body


def get_range(url: str, start: int, end_inclusive: int, budget: Budget, attempts: int = 4) -> bytes:
    """Range read with bounded retries on transient network errors (never on protocol violations)."""
    import time

    for i in range(attempts):
        try:
            return _get_range_once(url, start, end_inclusive, budget)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            if i == attempts - 1:
                raise RuntimeError(f"network failure after {attempts} attempts: {e}") from e
            time.sleep(2 * (i + 1))
    raise AssertionError("unreachable")


def _get_range_once(url: str, start: int, end_inclusive: int, budget: Budget) -> bytes:
    want = end_inclusive - start + 1
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored Range (status {r.status}) for {url}; stopping this path")
        cr = r.headers.get("Content-Range", "")
        m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
        if not m or int(m.group(1)) != start or int(m.group(2)) != end_inclusive:
            raise RuntimeError(f"unexpected Content-Range {cr!r} for {url}")
        declared = int(r.headers.get("Content-Length") or -1)
        if declared != want:
            raise RuntimeError(f"unexpected Content-Length {declared} (wanted {want})")
        budget.reserve(want)
        body = r.read(want + 1)
        if len(body) != want:
            raise RuntimeError("short/long range body")
        return body


def read_header(repo: str, rev: str, shard: str, budget: Budget) -> tuple[int, dict]:
    url = url_for(repo, rev, shard)
    n = struct.unpack("<Q", get_range(url, 0, 7, budget))[0]
    if n <= 0 or n > 50_000_000:
        raise RuntimeError(f"implausible header length {n} in {shard}")
    hdr = json.loads(get_range(url, 8, 8 + n - 1, budget))
    return n, hdr


def bf16_to_float(b: bytes) -> list[float]:
    out = []
    for i in range(0, len(b), 2):
        (u,) = struct.unpack("<H", b[i : i + 2])
        (f,) = struct.unpack("<f", struct.pack("<I", u << 16))
        out.append(f)
    return out


def family(name: str) -> str:
    return re.sub(r"\.\d+\.", ".{i}.", re.sub(r"\.\d+\.", ".{i}.", name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-mb", type=float, default=100.0)
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(int(a.budget_mb * 1_000_000))
    today = _dt.date.today().isoformat()
    headers_by_model: dict[str, dict] = {}

    for key in a.models.split(","):
        repo, rev = MODELS[key]
        idx = json.loads(get_small(url_for(repo, rev, "model.safetensors.index.json"), budget))
        shards = sorted(set(idx["weight_map"].values()))
        table = {}
        shard_info = []
        for s in shards:
            n, hdr = read_header(repo, rev, s, budget)
            meta = hdr.pop("__metadata__", None)
            shard_info.append({"file": s, "headerBytes": n, "metadata": meta, "tensors": len(hdr)})
            for t, v in hdr.items():
                table[t] = {"shard": s, "dtype": v["dtype"], "shape": v["shape"], "offsets": v["data_offsets"], "headerBytes": n}
            print(f"[{key}] {s}: {len(hdr)} tensors (header {n} B); budget used {budget.used/1e6:.2f} MB", file=sys.stderr)
        # cross-check index vs headers
        missing = [t for t in idx["weight_map"] if t not in table]
        wrong_shard = [t for t, s in idx["weight_map"].items() if t in table and table[t]["shard"] != s]
        total_bytes = 0
        total_elems = 0
        fams: dict[str, dict] = {}
        for t, v in table.items():
            ne = 1
            for d in v["shape"]:
                ne *= d
            nb = v["offsets"][1] - v["offsets"][0]
            if nb != ne * DTYPE_BYTES[v["dtype"]]:
                raise RuntimeError(f"offset/shape mismatch for {t}")
            total_bytes += nb
            total_elems += ne
            f = family(t)
            e = fams.setdefault(f, {"family": f, "count": 0, "dtype": set(), "shape": set(), "elements": 0, "shards": set()})
            e["count"] += 1
            e["dtype"].add(v["dtype"])
            e["shape"].add(tuple(v["shape"]))
            e["elements"] += ne
            e["shards"].add(v["shard"])
        catalog = {
            "model": repo,
            "revision": rev,
            "retrieved": today,
            "method": "HTTP Range reads of safetensors length prefix + JSON header only",
            "shards": shard_info,
            "tensorCount": len(table),
            "indexTensorCount": len(idx["weight_map"]),
            "indexTotalSize": idx.get("metadata", {}).get("total_size"),
            "headerTotalTensorBytes": total_bytes,
            "headerTotalElements": total_elems,
            "indexMissingFromHeaders": missing[:20],
            "indexShardMismatches": wrong_shard[:20],
            "families": [
                {
                    "family": e["family"],
                    "count": e["count"],
                    "dtype": sorted(e["dtype"]),
                    "shapes": [list(s) for s in sorted(e["shape"])],
                    "elements": e["elements"],
                    "shards": sorted(e["shards"]),
                }
                for e in sorted(fams.values(), key=lambda x: x["family"])
            ],
        }
        if key == "qwen3-30b-a3b":
            # per-layer shard placement of each expert matrix, compact: layer -> [shard numbers per expert] for gate/up/down
            place = {}
            for t, v in table.items():
                m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", t)
                if m:
                    L, E, w = int(m.group(1)), int(m.group(2)), m.group(3)
                    num = int(re.search(r"model-(\d+)-of", v["shard"]).group(1))
                    place.setdefault(L, {"gate": [0] * 128, "up": [0] * 128, "down": [0] * 128})[w][E] = num
            catalog["expertShardPlacement"] = {str(k): place[k] for k in sorted(place)}
        # per-layer shard of non-expert tensors
        layer_shards = {}
        for t, v in table.items():
            m = re.match(r"(?:model\.(?:language_model\.)?)layers\.(\d+)\.", t)
            if m and ".experts." not in t:
                num = int(re.search(r"model-(\d+)-of", v["shard"]).group(1))
                layer_shards.setdefault(int(m.group(1)), set()).add(num)
        catalog["layerShards"] = {str(k): sorted(layer_shards[k]) for k in sorted(layer_shards)}
        catalog["nonLayerTensors"] = {
            t: {"shard": v["shard"], "dtype": v["dtype"], "shape": v["shape"]}
            for t, v in table.items()
            if not re.search(r"layers\.\d+\.", t) and "ngram_embedding.shard_" not in t and not t.startswith("model.visual.blocks")
        }
        (out / f"headers-{key}.json").write_text(json.dumps(table))
        (out / f"catalog-{key}.json").write_text(json.dumps(catalog, indent=1))
        headers_by_model[key] = table
        print(f"[{key}] tensors={len(table)} elements={total_elems} bytes={total_bytes} index_total={catalog['indexTotalSize']}", file=sys.stderr)

    samples = []
    if not a.no_samples:
        for key, tname, r0, c0 in SAMPLE_TILES:
            if key not in headers_by_model:
                continue
            repo, rev = MODELS[key]
            v = headers_by_model[key][tname]
            if v["dtype"] != "BF16":
                print(f"skip {tname}: dtype {v['dtype']}", file=sys.stderr)
                continue
            shape = v["shape"]
            url = url_for(repo, rev, v["shard"])
            base = 8 + v["headerBytes"] + v["offsets"][0]
            raw = b""
            rows = []
            if len(shape) == 1:
                cols = min(TILE, shape[0] - c0)
                start = base + c0 * 2
                b = get_range(url, start, start + cols * 2 - 1, budget)
                raw += b
                rows.append(bf16_to_float(b))
                idx_rng = {"cols": [c0, c0 + cols - 1]}
            else:
                ncols = shape[1]
                cols = min(TILE, ncols - c0)
                for r in range(r0, min(r0 + TILE, shape[0])):
                    start = base + (r * ncols + c0) * 2
                    b = get_range(url, start, start + cols * 2 - 1, budget)
                    raw += b
                    rows.append(bf16_to_float(b))
                idx_rng = {"rows": [r0, r0 + len(rows) - 1], "cols": [c0, c0 + cols - 1]}
            samples.append(
                {
                    "id": f"{key}:{tname}:{r0}:{c0}",
                    "evidence": "actual-checkpoint-sample",
                    "modelKey": key,
                    "model": repo,
                    "revision": rev,
                    "shard": v["shard"],
                    "tensor": tname,
                    "logicalShape": shape,
                    "dtype": v["dtype"],
                    "index": idx_rng,
                    "conversion": "BF16 bit pattern widened exactly to float32 (no rounding, clipping or downsampling)",
                    "values": rows,
                    "rawSha256": hashlib.sha256(raw).hexdigest(),
                    "rawBytes": len(raw),
                    "byteRange": "per-row HTTP Range reads at 8 + header_len + data_offset + (row*ncols+col)*2",
                    "retrieved": today,
                    "license": "Apache-2.0 (model repository license)",
                    "note": "A tiny tile of stored weights. Not a complete model, not an activation, not an inference trace.",
                }
            )
            print(f"sample {tname}: {len(raw)} bytes sha256={samples[-1]['rawSha256'][:12]}", file=sys.stderr)
    (out / "samples.json").write_text(json.dumps({"budgetBytes": budget.limit, "bytesUsed": budget.used, "samples": samples}, indent=1))
    print(f"TOTAL bytes transferred (bodies): {budget.used}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, urllib.error.URLError) as e:
        print(f"EXTRACTION STOPPED: {e}", file=sys.stderr)
        sys.exit(2)
