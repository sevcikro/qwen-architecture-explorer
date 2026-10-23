"""Build one Jupyter notebook per course example (examples/python -> notebooks/), ready for Google Colab.

Each notebook has: a title and explanation, an install cell when the example needs Hugging Face libraries,
the example's code, and its recorded output from src/content/code-outputs.json (so it reads well before running).

Usage:  python -I scripts/make_notebooks.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "python"
NB = ROOT / "notebooks"
SITE = "https://happy-bamboo-w7dk.here.now/"
REPO = "sevcikro/qwen-architecture-explorer"
CHAPTER = {
    "01_neuron.py": "neuron", "02_vectors.py": "vectors", "03_softmax_xent.py": "probability", "04_tokenizer.py": "tokens",
    "05_embedding_rmsnorm.py": "residual", "06_attention.py": "attention", "07_gqa_rope.py": "heads", "08_ffn_moe.py": "moe",
    "09_tiny_transformer.py": "forward", "10_backprop.py": "backprop", "11_train_tiny_lm.py": "training-llm",
    "12_hf_generate.py": "journey", "13_hf_inspect.py": "hf", "14_safetensors.py": "files", "15_quantize.py": "serving",
    "16_api_client.py": "journey",
}
INSTALL = {
    "04_tokenizer.py": '%pip install -q "transformers>=4.56"',
    "12_hf_generate.py": '%pip install -q "transformers>=4.56"',
    "13_hf_inspect.py": '%pip install -q "transformers>=4.56"',
    "14_safetensors.py": "%pip install -q huggingface_hub safetensors",
}


def lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def main() -> None:
    outputs = json.loads((ROOT / "src" / "content" / "code-outputs.json").read_text(encoding="utf-8"))
    env = outputs["_env"]
    NB.mkdir(exist_ok=True)
    made = []
    for f in sorted(EX.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        doc = src.split('"""')[1].strip() if src.startswith('"""') else ""
        title = doc.split("\n")[0].rstrip(".")
        body = src.split('"""', 2)[2].lstrip("\n") if src.startswith('"""') else src
        chapter = CHAPTER[f.name]
        nb_name = f.stem + ".ipynb"
        md = [
            f"# {title}\n", "\n",
            f"Part of the free interactive course **Qwen Atlas** — chapter: [{chapter}]({SITE}#/learn/{chapter}).\n", "\n",
            "```text\n", *lines("\n".join(doc.split("\n")[1:]).strip() + "\n"), "```\n", "\n",
            f"The output under the code cell was recorded on {env['date']} with Python {env['python']}, numpy {env['numpy']}, "
            f"torch {env['torch']} and transformers {env['transformers']} on a {env['device']}. Run the cell to reproduce it.\n",
        ]
        cells = [{"cell_type": "markdown", "metadata": {}, "source": md}]
        if f.name in INSTALL:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [INSTALL[f.name]]})
        rec = outputs.get(f.name, {})
        code_outputs = [{"name": "stdout", "output_type": "stream", "text": lines(rec["stdout"] + "\n")}] if rec.get("ran") else []
        cells.append({"cell_type": "code", "execution_count": 1 if code_outputs else None, "metadata": {}, "outputs": code_outputs, "source": lines(body.rstrip("\n"))})
        if not rec.get("ran"):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": ["This example needs a running model server and was not executed when the course was built.\n"]})
        nb = {
            "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                         "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"{f.stem.replace('_', '-')}-{i}"
        (NB / nb_name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        made.append(nb_name)
    readme = ["# Course notebooks\n", "\n", "Generated from `examples/python` by `scripts/make_notebooks.py`. Open one in Google Colab:\n", "\n"]
    for n in made:
        readme.append(f"- [{n}](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{n})\n")
    (NB / "README.md").write_text("".join(readme), encoding="utf-8")
    print("wrote", len(made), "notebooks")


if __name__ == "__main__":
    main()
