"""
LNN Pruner Evaluation & Comparison — Colab Notebook
=====================================================
Bu script eğitilmiş S5a/S5b modellerini tüm diğer pruner'larla karşılaştırır.
Her `# %%` bloğunu Colab'da ayrı bir hücreye yapıştırın.

Metrikler:
  - Faithfulness (LLM-as-Judge)
  - Exact Match / Token F1
  - Compression Ratio
  - Supporting Fact Coverage (NEW)
  - Noise Ratio (NEW)
  - Coverage-Noise F1 (NEW)
"""

# %% [markdown]
# # 📊 LNN Pruner Evaluation & Comparison Table
# Eğitilmiş S5a ve S5b modellerini B1–B3, S1–S4 ile karşılaştırıyoruz.

# %% Setup
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("datasets")
install("sentence-transformers")
install("ncps")
install("torch")
install("rank_bm25")
install("tqdm")
install("matplotlib")
install("pandas")
install("transformers")
install("accelerate")

REPO_DIR = "/content/CENG467_Final"
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %% Load Data
from datasets import load_dataset

ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
val_data = ds["validation"]

def convert_sample(sample):
    context = list(zip(sample["context"]["title"], sample["context"]["sentences"]))
    return {
        "id": sample["id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "context": context,
        "supporting_facts": {
            "title": sample["supporting_facts"]["title"],
            "sent_idx": sample["supporting_facts"]["sent_idx"],
        },
        "type": sample["type"],
        "level": sample["level"],
    }

# Use samples AFTER training set to avoid data leakage
EVAL_START = 600
N_EVAL = 50  # Start with 50, scale to 200 later

all_samples = [convert_sample(val_data[i]) for i in range(len(val_data))]
eval_samples = all_samples[EVAL_START : EVAL_START + N_EVAL]
print(f"✅ Loaded {len(eval_samples)} evaluation samples (indices {EVAL_START}–{EVAL_START + N_EVAL}).")

# %% Import pipeline components
import torch
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.retrieval.bm25_retriever import BM25Retriever
from src.pruning import PRUNER_REGISTRY
from src.evaluation.coverage_noise import (
    supporting_fact_coverage,
    noise_ratio,
    coverage_noise_f1,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")

# %% Define all pruner configurations
PRUNER_CONFIGS = {
    "no_pruning":       {"label": "B1: No Pruning",       "class": "no_pruning"},
    "naive_truncation": {"label": "B2: Naive Truncation",  "class": "naive_truncation"},
    "recomp":           {"label": "B3: RECOMP",            "class": "recomp"},
    "provence":         {"label": "S2: Provence",          "class": "provence"},
    "history_pruning":  {"label": "S3: History Pruning",   "class": "history_pruning"},
    "combined":         {"label": "S4: Combined",          "class": "combined"},
    "lnn_sentence":     {"label": "S5a: LNN Sentence",    "class": "lnn_sentence"},
    "lnn_token":        {"label": "S5b: LNN Token",        "class": "lnn_token"},
}

# %% [markdown]
# ## 🔬 Pruning-Only Evaluation (No LLM Required)
# Coverage, noise, compression metrics ile tüm pruner'ları karşılaştırıyoruz.
# Bu adım LLM gerektirmez — sadece pruning kalitesini ölçer.

# %% Run pruning-only evaluation
results = {}

for name, config in PRUNER_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"  Evaluating: {config['label']}")
    print(f"{'='*60}")

    pruner = PRUNER_REGISTRY[config["class"]]()
    retriever = BM25Retriever()

    metrics = {
        "coverage": [], "noise_ratio": [], "cn_f1": [],
        "compression_ratio": [], "latency_s": [],
    }

    for sample in tqdm(eval_samples, desc=f"  {config['label']}"):
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        retrieved = retriever.retrieve(sample["question"], k=5)

        input_tokens = sum(len(p["passage"].split()) for p in retrieved)

        t0 = time.time()
        pruned = pruner.prune(retrieved, sample["question"], max_tokens=512)
        elapsed = time.time() - t0

        output_tokens = sum(len(p["passage"].split()) for p in pruned)
        comp = output_tokens / max(input_tokens, 1)

        cov = supporting_fact_coverage(pruned, sample)
        noise = noise_ratio(pruned, sample)
        cn_f1 = coverage_noise_f1(cov, noise)

        metrics["coverage"].append(cov)
        metrics["noise_ratio"].append(noise)
        metrics["cn_f1"].append(cn_f1)
        metrics["compression_ratio"].append(comp)
        metrics["latency_s"].append(elapsed)

    n = len(eval_samples)
    results[name] = {
        "label": config["label"],
        "coverage": np.mean(metrics["coverage"]),
        "noise_ratio": np.mean(metrics["noise_ratio"]),
        "cn_f1": np.mean(metrics["cn_f1"]),
        "compression_ratio": np.mean(metrics["compression_ratio"]),
        "latency_s": np.mean(metrics["latency_s"]),
    }

    r = results[name]
    print(f"  Coverage     : {r['coverage']:.3f}")
    print(f"  Noise Ratio  : {r['noise_ratio']:.3f}")
    print(f"  CN-F1        : {r['cn_f1']:.3f}")
    print(f"  Comp. Ratio  : {r['compression_ratio']:.3f}")
    print(f"  Latency (s)  : {r['latency_s']:.4f}")

# %% Comparison Table
import pandas as pd

rows = []
for name, r in results.items():
    rows.append({
        "Strategy": r["label"],
        "Coverage ↑": f"{r['coverage']:.3f}",
        "Noise ↓": f"{r['noise_ratio']:.3f}",
        "CN-F1 ↑": f"{r['cn_f1']:.3f}",
        "Comp. Ratio": f"{r['compression_ratio']:.3f}",
        "Latency (s)": f"{r['latency_s']:.4f}",
    })

df = pd.DataFrame(rows)
print("\n" + "=" * 90)
print("COMPARISON TABLE — Pruning Quality Metrics")
print("=" * 90)
print(df.to_string(index=False))
print("=" * 90)

# Save to CSV
os.makedirs("experiments/results", exist_ok=True)
df.to_csv("experiments/results/lnn_comparison.csv", index=False)
print("\n💾 Saved to experiments/results/lnn_comparison.csv")

# %% Visualization — Coverage vs Noise Trade-off
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

labels = [r["label"] for r in results.values()]
coverages = [r["coverage"] for r in results.values()]
noises = [r["noise_ratio"] for r in results.values()]
cn_f1s = [r["cn_f1"] for r in results.values()]
comp_ratios = [r["compression_ratio"] for r in results.values()]

colors = ["#6c757d"] * 3 + ["#0077b6"] * 2 + ["#f72585", "#4361ee"]
if len(labels) == 8:
    colors = ["#6c757d", "#6c757d", "#6c757d", "#0077b6", "#0077b6", "#0077b6",
              "#f72585", "#4361ee"]

# Plot 1: Coverage vs Noise scatter
ax = axes[0]
for i, (lab, cov, noise) in enumerate(zip(labels, coverages, noises)):
    c = colors[i] if i < len(colors) else "#333"
    short = lab.split(":")[0].strip()
    ax.scatter(noise, cov, s=120, c=c, zorder=5)
    ax.annotate(short, (noise, cov), textcoords="offset points",
                xytext=(8, 4), fontsize=9)
ax.set_xlabel("Noise Ratio ↓", fontsize=12)
ax.set_ylabel("Coverage ↑", fontsize=12)
ax.set_title("Coverage vs Noise Trade-off", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Plot 2: CN-F1 bar chart
ax = axes[1]
short_labels = [l.split(":")[0].strip() for l in labels]
bars = ax.bar(short_labels, cn_f1s, color=colors[:len(labels)], edgecolor="white")
ax.set_ylabel("Coverage-Noise F1 ↑", fontsize=12)
ax.set_title("Coverage-Noise F1 Score", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Plot 3: Compression Ratio
ax = axes[2]
ax.bar(short_labels, comp_ratios, color=colors[:len(labels)], edgecolor="white")
ax.set_ylabel("Compression Ratio", fontsize=12)
ax.set_title("Context Compression", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("experiments/results/lnn_comparison_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("📈 Comparison plots saved.")

# %% [markdown]
# ## 🤖 Full Pipeline Evaluation (with LLM — Optional)
# Faithfulness scoring icin Mistral-7B veya baska bir model lazim.
# Colab Pro (A100) gerektirebilir.

# %% Full pipeline with LLM (optional — uncomment to run)
"""
# Uncomment this cell if you have enough GPU VRAM for Mistral-7B

from src.generation.generator import RAGGenerator
from src.judge.judge import LLMJudge
from src.evaluation.metrics import exact_match, token_f1

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading {MODEL} ...")
generator = RAGGenerator(MODEL)
generator.load_model()
judge = LLMJudge(MODEL)
judge._model = generator._model
judge._tokenizer = generator._tokenizer
print("Model ready.")

full_results = {}

for name, config in PRUNER_CONFIGS.items():
    pruner = PRUNER_REGISTRY[config["class"]]()
    retriever = BM25Retriever()

    metrics = {
        "faithfulness": [], "em": [], "f1": [],
        "coverage": [], "noise_ratio": [], "cn_f1": [],
        "compression_ratio": [], "latency_s": [],
    }

    for sample in tqdm(eval_samples, desc=config["label"]):
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        retrieved = retriever.retrieve(sample["question"], k=5)

        input_tokens = sum(len(p["passage"].split()) for p in retrieved)
        pruned = pruner.prune(retrieved, sample["question"], max_tokens=512)
        output_tokens = sum(len(p["passage"].split()) for p in pruned)

        gen_result = generator.generate(sample["question"], pruned)
        judge_result = judge.score(gen_result["answer"], pruned)

        em = exact_match(gen_result["answer"], sample["answer"])
        f1 = token_f1(gen_result["answer"], sample["answer"])
        comp = output_tokens / max(input_tokens, 1)
        cov = supporting_fact_coverage(pruned, sample)
        noise = noise_ratio(pruned, sample)
        cn_f1_val = coverage_noise_f1(cov, noise)

        for k, v in [("faithfulness", judge_result["faithfulness"]),
                      ("em", em), ("f1", f1),
                      ("coverage", cov), ("noise_ratio", noise),
                      ("cn_f1", cn_f1_val),
                      ("compression_ratio", comp),
                      ("latency_s", gen_result["latency_s"])]:
            metrics[k].append(v)

    full_results[name] = {
        "label": config["label"],
        **{k: float(np.mean(v)) for k, v in metrics.items()},
    }

# Print full comparison
print("\\n" + "=" * 110)
print("FULL COMPARISON TABLE")
print("=" * 110)
print(f"{'Strategy':<26} {'Faith':>7} {'EM':>6} {'F1':>6} {'Cov':>6} "
      f"{'Noise':>6} {'CN-F1':>6} {'Comp':>6} {'Lat(s)':>8}")
print("-" * 110)
for name, r in full_results.items():
    print(f"{r['label']:<26} {r['faithfulness']:>7.3f} {r['em']:>6.3f} "
          f"{r['f1']:>6.3f} {r['coverage']:>6.3f} {r['noise_ratio']:>6.3f} "
          f"{r['cn_f1']:>6.3f} {r['compression_ratio']:>6.3f} "
          f"{r['latency_s']:>8.2f}")
print("=" * 110)

# Save full results
with open("experiments/results/full_comparison.json", "w") as f:
    json.dump(full_results, f, indent=2)
print("Saved to experiments/results/full_comparison.json")
"""

# %% [markdown]
# ## 📋 Sonuçlar
#
# | Metric | Açıklama |
# |--------|----------|
# | **Coverage ↑** | Gold supporting sentence'ların ne kadarı pruning sonrası kaldı |
# | **Noise Ratio ↓** | Kalan token'ların ne kadarı non-supporting passage'dan |
# | **CN-F1** | Coverage × Purity harmonic mean |
# | **Comp. Ratio** | Pruned tokens / Original tokens |
# | **Faithfulness** | LLM-as-Judge: atomic claim'lerin context'te desteklenme oranı |
#
# S5a ve S5b, LTC nöronlarının adaptif time constant'ları sayesinde
# token importance'ı zaman-bağımlı olarak modelleyebilir — bu da
# multi-turn conversation'larda daha iyi pruning kararları verir.
