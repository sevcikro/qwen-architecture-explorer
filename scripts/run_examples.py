"""Run every course example in examples/python and record its real output.

The recorded stdout is shown on the site next to the code, together with the library
versions used. Examples that need a server are skipped (listed in SKIP).

Usage (from the project root):
  QWEN_MODEL_DIR=/path/to/Qwen3-0.6B python -I scripts/run_examples.py --python /path/to/venv/python
"""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "python"
OUT = ROOT / "src" / "content" / "code-outputs.json"
SKIP = {"16_api_client.py": "needs a running OpenAI-compatible server; not executed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    model_dir = os.environ.get("QWEN_MODEL_DIR", "")
    versions = subprocess.run(
        [a.python, "-I", "-c", "import sys, numpy, torch, transformers; print(sys.version.split()[0], numpy.__version__, torch.__version__, transformers.__version__)"],
        capture_output=True, text=True, check=True).stdout.split()
    env_info = dict(zip(["python", "numpy", "torch", "transformers"], versions))
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    results.pop("_env", None)
    for f in sorted(EX.glob("*.py")):
        if a.only and a.only not in f.name:
            continue
        if f.name in SKIP:
            results[f.name] = {"ran": False, "note": SKIP[f.name]}
            continue
        args = [a.python, "-I", "-X", "utf8", str(f)]  # -I ignores PYTHONIOENCODING, so force UTF-8 mode
        if f.name == "14_safetensors.py":
            args.append(str(pathlib.Path(model_dir) / "model.safetensors"))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=str(EX), env=env, timeout=900)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise SystemExit(f"{f.name} failed with exit code {r.returncode}")
        out = r.stdout.replace(model_dir, "<QWEN_MODEL_DIR>") if model_dir else r.stdout
        results[f.name] = {"ran": True, "stdout": out.rstrip("\n")}
        print(f"ok  {f.name}  ({len(out.splitlines())} lines)")
    results["_env"] = {**env_info, "date": datetime.date.today().isoformat(), "device": "CPU"}
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
