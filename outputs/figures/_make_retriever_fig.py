"""Build the 3-way retriever comparison figure for the LNCS report.

Numbers are the no_pruning, 50-sample HotpotQA validation slice (indices
600-649) aggregates exported by notebooks 06 and 07.
"""
import matplotlib.pyplot as plt
import numpy as np

# retriever -> metrics (BM25 taken from the notebook-06 run)
data = {
    "BM25":          dict(recall=0.760, mrr=0.808, faith=0.770, em=0.100, f1=0.186, lat=3.78),
    "Dense (BGE)":   dict(recall=0.900, mrr=0.944, faith=0.714, em=0.100, f1=0.231, lat=2.75),
    "Cross-Encoder": dict(recall=0.850, mrr=0.947, faith=0.747, em=0.140, f1=0.261, lat=2.21),
}

metrics = [
    ("recall", "Recall@5"),
    ("mrr", "MRR"),
    ("faith", "Faithfulness"),
    ("f1", "Token F1"),
]
labels = list(data.keys())
colors = ["#4C78A8", "#F58518", "#54A24B"]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(8.2, 3.8))
for i, name in enumerate(labels):
    vals = [data[name][k] for k, _ in metrics]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels([t for _, t in metrics])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Retriever comparison on HotpotQA (no pruning, n=50)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", color="gray", alpha=0.15)
fig.tight_layout()
fig.savefig("outputs/figures/retriever_comparison_3way.png", dpi=160, bbox_inches="tight")
print("saved outputs/figures/retriever_comparison_3way.png")
