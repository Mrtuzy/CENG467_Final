import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup: ensure 'src/' is on sys.path so cross-imports work when
# scripts are invoked as  `python src/X.py`  from the repo root.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Çıktı kök dizini.  Öncelik sırası:
#   1) CENG467_BASE env değişkeni (lokal / sunucu koşumları için)
#   2) Colab Drive mount'u varsa  /content/drive/MyDrive/CENG_467
#   3) repo içinde  ./artifacts   (varsayılan, lokal)
# ---------------------------------------------------------------------------
def _resolve_base_dir() -> str:
    env_base = os.environ.get("CENG467_BASE")
    if env_base:
        return env_base
    colab_drive = "/content/drive/MyDrive/CENG_467"
    if os.path.isdir("/content/drive/MyDrive"):
        return colab_drive
    repo_root = os.path.dirname(_SRC_DIR)
    return os.path.join(repo_root, "artifacts")


DRIVE_BASE_DIR = _resolve_base_dir()
DATA_DIR       = os.path.join(DRIVE_BASE_DIR, "data")
MODEL_DIR      = os.path.join(DRIVE_BASE_DIR, "models")
OUTPUT_DIR     = os.path.join(DRIVE_BASE_DIR, "outputs")
FIGURES_DIR    = os.path.join(DRIVE_BASE_DIR, "figures")

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
PROXY_MODEL_NAME   = "distilgpt2"
SBERT_MODEL_NAME   = "all-MiniLM-L6-v2"
# Öğretmen = hem önem etiketlemesinde (cevap-koşullu ΔNLL) hem de
# değerlendirmede üretici LLM olarak kullanılır.  Hafif, gated değil ve hızlı:
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Eski (ağır, gated) öğretmen: "mistralai/Mistral-7B-Instruct-v0.2"

# ---------------------------------------------------------------------------
# CfC hyper-parameters
# ---------------------------------------------------------------------------
SBERT_DIM     = 384          # sentence-transformer output dimension
# --- entropi → Δt eşlemesi ---
# Değerler simulations/cfc_proof kanıt-suite'inden gelir (bkz. GUIDE.md §5).
DELTA_T_MIN   = 0.05         # S2: kurtarmayı maksimize eden Δt_min (eski varsayılan 0.1)
BETA          = 1.0          # S2: optimum β (config zaten doğrulandı)
DT_MAPPING    = "rank"       # S5: aykırı surprisal'a en dayanıklı eşleme
                             #     {"linear": Δt_min+β·S, "rank": Δt_min+β·rank(S)/N}
CFC_UNITS     = 64           # hidden size of CfC
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
EPOCHS        = 20           # 30→20: en iyi model val_loss ile kaydedilir, 20 yeter
TAU           = 0.5          # pruning threshold

# Dataset size limits — A100'de tüm pipeline ~2 saatte bitsin diye ayarlı.
# Daha fazla veri istersen bu üçünü büyüt (süre lineer artar).
MAX_TRAIN_SAMPLES     = 3000   # teacher labeling + build_inputs maliyetini belirler
MAX_EVAL_SAMPLES      = 500    # evaluate.py: örnek başına 4 generation
ABLATION_MAX_SAMPLES  = 200    # ablation.py: örnek × TAU sayısı kadar generation

def get_args():
    """Parse CLI arguments shared by every script."""
    parser = argparse.ArgumentParser(description="Entropy-Driven CfC Pruning")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick dry-run with minimal data")
    return parser.parse_known_args()[0]
