"""
LNN Pruner Training — Colab Notebook
=====================================
Bu scripti Colab'da çalıştırmak için her `# %%` bloğunu ayrı bir hücreye yapıştırın.
Veya: Runtime > Run all ile tamamını çalıştırabilirsiniz.

Sıra:
  1. Setup & Install
  2. Data Loading
  3. Train S5a (Sentence-level LNN)
  4. Train S5b (Token-level LNN)
  5. Training Curves Visualization
"""

# %% [markdown]
# # 🧠 LNN-Based Pruner Training for RAG Faithfulness
# **Liquid Time-Constant (LTC) Networks** ile sentence-level ve token-level
# pruning modellerini eğitiyoruz.
#
# Tez: *Token'ların önemi zaman içinde değişir — LNN'ler bu dinamiği
# ODE-tabanlı adaptif time constant'ları sayesinde doğal olarak modeller.*

# %% Setup & Install
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("datasets")
install("sentence-transformers")
install("ncps")
install("torch")
install("tqdm")
install("matplotlib")

print("✅ All packages installed.")

# %% Mount Drive (optional — if repo is on Drive)
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/CENG467_Final

# %% Clone or setup repo
import os

REPO_DIR = "/content/CENG467_Final"

if not os.path.exists(REPO_DIR):
    print("📂 Repo not found. Creating working directory ...")
    os.makedirs(REPO_DIR, exist_ok=True)
    # If you have a GitHub repo, uncomment:
    # !git clone https://github.com/YOUR_REPO/CENG467_Final.git {REPO_DIR}

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %% Load HotpotQA Data
from datasets import load_dataset
import json

print("📥 Loading HotpotQA (distractor) ...")
ds = load_dataset("hotpot_qa", "distractor")

# Use validation split
val_data = ds["validation"]

# Convert to list of dicts with proper structure
def convert_sample(sample):
    """Convert HuggingFace sample to our pipeline format."""
    context = list(zip(sample["context"]["title"], sample["context"]["sentences"]))
    return {
        "id": sample["id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "context": context,
        "supporting_facts": {
            "title": sample["supporting_facts"]["title"],
            "sent_id": sample["supporting_facts"]["sent_id"],
        },
        "type": sample["type"],
        "level": sample["level"],
    }

samples = [convert_sample(val_data[i]) for i in range(len(val_data))]
print(f"✅ Loaded {len(samples)} validation samples.")

# Split: 500 train, 100 val for LNN training
# (We use validation split for LNN training since we don't fine-tune the LLM)
TRAIN_SIZE = 500
VAL_SIZE = 100

train_samples = samples[:TRAIN_SIZE]
val_samples = samples[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
eval_samples = samples[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + 200]

print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}, Eval: {len(eval_samples)}")

# %% Initialize Shared Encoder
import torch

# --- Paste src modules if not available as package ---
# If the repo is properly cloned, these imports work directly.
# Otherwise, copy the src/ files to Colab.
try:
    from src.pruning.lnn_utils import SentenceEncoder, MultiTurnSimulator
    from src.pruning.lnn_trainer import train_sentence_model, train_token_model
    print("✅ Imported from src package.")
except ImportError:
    print("⚠️  src package not found — please upload src/ directory to Colab.")
    print("   Or clone the repo: !git clone <repo_url> /content/CENG467_Final")
    raise

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Pre-load encoder (shared between S5a and S5b)
encoder = SentenceEncoder(device=device)
encoder.load()
print(f"✅ Sentence encoder loaded (dim={encoder.embedding_dim}).")

# %% [markdown]
# ## 📐 Train S5a: Sentence-Level LNN Pruner
# Her cümle sıralı olarak LTC cell'den geçirilir.
# Hidden state conversation turn'leri arasında taşınır.

# %% Train S5a
import os

os.makedirs("models", exist_ok=True)

history_s5a = train_sentence_model(
    train_samples=train_samples,
    val_samples=val_samples,
    epochs=20,
    lr=1e-3,
    turns_per_conv=3,
    decay_factor=0.5,
    device=device,
    save_path="models/lnn_sentence.pt",
    encoder=encoder,
)

print("\n✅ S5a training complete!")

# %% [markdown]
# ## 🔤 Train S5b: Token-Level LNN Pruner
# Token sequence'i LTC-RNN'den geçirilir.
# Per-token importance score üretilir.

# %% Train S5b
history_s5b = train_token_model(
    train_samples=train_samples,
    val_samples=val_samples,
    epochs=20,
    lr=1e-3,
    turns_per_conv=3,
    decay_factor=0.5,
    device=device,
    save_path="models/lnn_token.pt",
    encoder=encoder,
)

print("\n✅ S5b training complete!")

# %% Training Curves
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# S5a curves
ax = axes[0]
ax.plot(history_s5a["train_loss"], label="Train Loss", color="#4361ee", linewidth=2)
if history_s5a["val_loss"]:
    ax.plot(history_s5a["val_loss"], label="Val Loss", color="#f72585", linewidth=2)
ax.set_title("S5a: Sentence-Level LNN", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# S5b curves
ax = axes[1]
ax.plot(history_s5b["train_loss"], label="Train Loss", color="#4361ee", linewidth=2)
if history_s5b["val_loss"]:
    ax.plot(history_s5b["val_loss"], label="Val Loss", color="#f72585", linewidth=2)
ax.set_title("S5b: Token-Level LNN", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("📈 Training curves saved to models/training_curves.png")

# %% Save training history
import json

with open("models/training_history.json", "w") as f:
    json.dump({
        "s5a": history_s5a,
        "s5b": history_s5b,
    }, f, indent=2)

print("💾 Training history saved to models/training_history.json")

# %% [markdown]
# ## ✅ Eğitim Tamamlandı
#
# Model dosyaları:
# - `models/lnn_sentence.pt` — S5a ağırlıkları
# - `models/lnn_token.pt` — S5b ağırlıkları
# - `models/training_curves.png` — Loss grafikleri
# - `models/training_history.json` — Epoch-level loss değerleri
#
# Sonraki adım: `02_lnn_evaluation_colab.py` ile evaluation yapın.
