"""Generate the two retriever-comparison Colab notebooks (A100).

Run:  python notebooks/_build_retrieval_notebooks.py
Produces:
  notebooks/06_dense_retriever.ipynb
  notebooks/07_cross_encoder_retriever.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# --------------------------------------------------------------------------- #
# Shared cells
# --------------------------------------------------------------------------- #
SETUP = """\
# === Environment setup (Colab, A100) ===
import os, sys, subprocess

REPO_URL = "https://github.com/Mrtuzy/CENG467_Final.git"
REPO_DIR = "/content/CENG467_Final"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Pull latest (so the new retrievers + runner changes are present)
subprocess.run(["git", "pull", "--ff-only"], check=False)

!pip install -q -r requirements.txt
import nltk
nltk.download("punkt", quiet=True)
print("Setup done. CWD:", os.getcwd())
"""

HF = """\
# === Hugging Face auth (Mistral-7B is gated -> needs an accepted-license token) ===
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab secrets.")
except Exception:
    from getpass import getpass
    os.environ["HF_TOKEN"] = getpass("Enter your Hugging Face token: ")
"""

CONFIG_TMPL = """\
# === Experiment configuration ===
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # also used as the LLM-as-Judge
JUDGE_MODEL     = GENERATOR_MODEL                        # shared weights -> one 7B load

N_SAMPLES   = 50      # HotpotQA validation samples to evaluate
VAL_START   = 600     # start index in the validation split (matches Notebook 5 slice)
K_RETRIEVE  = 5       # top-k passages kept by the retriever

# Hold the pruner fixed so differences are attributable to the RETRIEVER.
# Add "recomp" to also see the retriever x pruner interaction (doubles runtime).
PRUNERS = ["no_pruning"]

# The new retriever this notebook studies, plus BM25 as the in-notebook anchor.
NEW_RETRIEVER_NAME = "{new_name}"

OUT_DIR = "experiments/results/retrieval"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
print("Config ready.")
"""

DATA = """\
# === Load the HotpotQA validation slice ===
from src.data.loader import load_hotpotqa

# load_hotpotqa returns samples from the start of the split; slice to VAL_START.
all_samples = load_hotpotqa("validation", n_samples=VAL_START + N_SAMPLES)
samples = all_samples[VAL_START:VAL_START + N_SAMPLES]
print(f"Loaded {len(samples)} samples (indices {VAL_START}..{VAL_START + N_SAMPLES - 1}).")
print("Example:", samples[0]["question"][:90])
"""

RUN_TMPL = """\
# === Run the pipeline: BM25 anchor + {new_label} ===
import json
from src.evaluation.runner import EvaluationRunner
from src.retrieval.bm25_retriever import BM25Retriever
{import_line}

def make_retrievers():
    # Re-instantiated per retriever so model weights load once and are cached.
    return {{
        "bm25": BM25Retriever(),
        NEW_RETRIEVER_NAME: {new_ctor},
    }}

retrievers = make_retrievers()
aggregates = []

for ret_name, ret in retrievers.items():
    for pruner in PRUNERS:
        out_path = f"{{OUT_DIR}}/{{ret_name}}_{{pruner}}.jsonl"
        runner = EvaluationRunner(
            pruner_name=pruner,
            generator_model=GENERATOR_MODEL,
            judge_model=JUDGE_MODEL,
            output_path=out_path,
            retriever=ret,
            retriever_name=ret_name,
        )
        agg = runner.run(samples, k_retrieve=K_RETRIEVE)
        aggregates.append(agg)
        print(f"[done] {{ret_name}} / {{pruner}}: "
              f"recall={{agg['retrieval_recall_mean']:.3f}} "
              f"mrr={{agg['retrieval_mrr_mean']:.3f}} "
              f"faith={{agg['faithfulness_mean']:.3f}} "
              f"EM={{agg['em_mean']:.3f}} F1={{agg['f1_mean']:.3f}}")

agg_path = f"{{OUT_DIR}}/aggregates_{{NEW_RETRIEVER_NAME}}.json"
with open(agg_path, "w") as f:
    json.dump(aggregates, f, indent=2)
print("Saved aggregates ->", agg_path)
"""

TABLE = """\
# === Comparison table ===
import pandas as pd
df = pd.DataFrame(aggregates)
cols = ["retriever", "pruner", "retrieval_recall_mean", "retrieval_mrr_mean",
        "faithfulness_mean", "em_mean", "f1_mean", "latency_mean_s"]
df = df[cols].round(3)
df.columns = ["Retriever", "Pruner", "Recall@k", "MRR", "Faithfulness",
              "EM", "F1", "Latency(s)"]
df
"""

PLOT_TMPL = """\
# === Bar chart: {new_label} vs BM25 (pruner = first in PRUNERS) ===
import matplotlib.pyplot as plt
import numpy as np

p0 = PRUNERS[0]
sub = [a for a in aggregates if a["pruner"] == p0]
order = ["bm25", NEW_RETRIEVER_NAME]
sub = sorted(sub, key=lambda a: order.index(a["retriever"]))
labels = [a["retriever"] for a in sub]

metrics = [
    ("retrieval_recall_mean", "Recall@k"),
    ("retrieval_mrr_mean", "MRR"),
    ("faithfulness_mean", "Faithfulness"),
    ("f1_mean", "Token F1"),
]
fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
colors = ["#4C78A8", "#F58518"]
for ax, (key, title) in zip(axes, metrics):
    vals = [a[key] for a in sub]
    ax.bar(labels, vals, color=colors[:len(labels)])
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.25 + 1e-6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{{v:.3f}}", ha="center", va="bottom", fontsize=9)
fig.suptitle(f"{new_label} vs BM25  (pruner={{p0}}, n={{N_SAMPLES}})", y=1.04)
fig.tight_layout()
fig_path = "outputs/figures/retrieval_{new_name}_vs_bm25.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print("Saved figure ->", fig_path)
"""


def build_notebook(new_name, new_label, import_line, new_ctor, extra_cells=None):
    cells = [
        md(f"# {new_label} vs BM25 — Retriever Comparison\n\n"
           "Drop-in replacement for the BM25 retriever used in our RAG pipeline.\n"
           "Everything else (generator, judge, pruner) is held fixed, so any\n"
           "difference is attributable to the **retriever**.\n\n"
           "**Runtime:** Colab GPU, A100 recommended. Mistral-7B is loaded in 4-bit.\n"),
        md("## 1. Setup"),
        code(SETUP),
        code(HF),
        md("## 2. Configuration"),
        code(CONFIG_TMPL.format(new_name=new_name)),
        md("## 3. Load data"),
        code(DATA),
        md(f"## 4. Run pipeline ({new_label} + BM25)\n\n"
           "Retrieval-quality metrics (Recall@k, MRR vs HotpotQA gold supporting\n"
           "passages) are computed alongside the downstream faithfulness/EM/F1.\n"),
        code(RUN_TMPL.format(new_label=new_label, import_line=import_line, new_ctor=new_ctor)),
        md("## 5. Results"),
        code(TABLE),
        code(PLOT_TMPL.format(new_name=new_name, new_label=new_label)),
    ]
    if extra_cells:
        cells.extend(extra_cells)
    return notebook(cells)


# --------------------------------------------------------------------------- #
# Notebook 06 — Dense bi-encoder
# --------------------------------------------------------------------------- #
nb06 = build_notebook(
    new_name="dense",
    new_label="Dense Bi-Encoder (BGE)",
    import_line="from src.retrieval.dense_retriever import DenseRetriever",
    new_ctor='DenseRetriever(model_name="BAAI/bge-small-en-v1.5")',
)

# --------------------------------------------------------------------------- #
# Notebook 07 — Cross-encoder reranker + 3-way merge
# --------------------------------------------------------------------------- #
THREEWAY = """\
# === 3-way comparison: BM25 vs Dense vs Cross-Encoder ===
# Loads the Dense aggregates from Notebook 06 if available; otherwise shows the
# two retrievers from this notebook only.
import json, glob
import pandas as pd
import matplotlib.pyplot as plt

rows = list(aggregates)  # bm25 + cross_encoder from this run
dense_path = f"{OUT_DIR}/aggregates_dense.json"
if os.path.exists(dense_path):
    with open(dense_path) as f:
        dense_aggs = json.load(f)
    # keep only the new (dense) rows; bm25 is already present from this run
    rows += [a for a in dense_aggs if a["retriever"] != "bm25"]
    print("Merged Dense aggregates from Notebook 06.")
else:
    print("Notebook 06 aggregates not found; run 06_dense_retriever.ipynb first "
          "for the full 3-way comparison.")

p0 = PRUNERS[0]
rows = [a for a in rows if a["pruner"] == p0]
order = ["bm25", "dense", "cross_encoder"]
rows = sorted(rows, key=lambda a: order.index(a["retriever"]) if a["retriever"] in order else 99)

mdf = pd.DataFrame(rows)[
    ["retriever", "retrieval_recall_mean", "retrieval_mrr_mean",
     "faithfulness_mean", "em_mean", "f1_mean", "latency_mean_s"]
].round(3)
mdf.columns = ["Retriever", "Recall@k", "MRR", "Faithfulness", "EM", "F1", "Latency(s)"]
display(mdf)

# grouped bar chart
metrics = [("retrieval_recall_mean", "Recall@k"), ("retrieval_mrr_mean", "MRR"),
           ("faithfulness_mean", "Faithfulness"), ("f1_mean", "Token F1")]
labels = [a["retriever"] for a in rows]
x = range(len(metrics))
width = 0.8 / max(len(rows), 1)
fig, ax = plt.subplots(figsize=(10, 4.2))
palette = ["#4C78A8", "#F58518", "#54A24B"]
for i, a in enumerate(rows):
    vals = [a[k] for k, _ in metrics]
    ax.bar([xi + i * width for xi in x], vals, width,
           label=a["retriever"], color=palette[i % len(palette)])
ax.set_xticks([xi + width * (len(rows) - 1) / 2 for xi in x])
ax.set_xticklabels([t for _, t in metrics])
ax.set_title(f"Retriever comparison (pruner={p0}, n={N_SAMPLES})")
ax.legend()
fig.tight_layout()
fig.savefig("outputs/figures/retrieval_3way_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> outputs/figures/retrieval_3way_comparison.png")
"""

nb07 = build_notebook(
    new_name="cross_encoder",
    new_label="Cross-Encoder Reranker",
    import_line="from src.retrieval.cross_encoder_retriever import CrossEncoderRetriever",
    new_ctor='CrossEncoderRetriever(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")',
    extra_cells=[
        md("## 6. Three-way comparison (BM25 / Dense / Cross-Encoder)\n\n"
           "Run `06_dense_retriever.ipynb` first (same `N_SAMPLES`/`VAL_START`) so its\n"
           "Dense aggregates are available to merge here."),
        code(THREEWAY),
    ],
)

(HERE / "06_dense_retriever.ipynb").write_text(json.dumps(nb06, indent=1))
(HERE / "07_cross_encoder_retriever.ipynb").write_text(json.dumps(nb07, indent=1))
print("Wrote 06_dense_retriever.ipynb and 07_cross_encoder_retriever.ipynb")
