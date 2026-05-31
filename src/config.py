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
# Google Drive paths  (mount point: /content/drive)
# ---------------------------------------------------------------------------
DRIVE_BASE_DIR = "/content/drive/MyDrive/CENG_467"
DATA_DIR       = os.path.join(DRIVE_BASE_DIR, "data")
MODEL_DIR      = os.path.join(DRIVE_BASE_DIR, "models")
OUTPUT_DIR     = os.path.join(DRIVE_BASE_DIR, "outputs")
FIGURES_DIR    = os.path.join(DRIVE_BASE_DIR, "figures")

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
PROXY_MODEL_NAME   = "distilgpt2"
SBERT_MODEL_NAME   = "all-MiniLM-L6-v2"
TEACHER_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# CfC hyper-parameters
# ---------------------------------------------------------------------------
SBERT_DIM     = 384          # sentence-transformer output dimension
DELTA_T_MIN   = 0.1
BETA          = 1.0
CFC_UNITS     = 64           # hidden size of CfC
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
EPOCHS        = 30
TAU           = 0.5          # pruning threshold

def get_args():
    """Parse CLI arguments shared by every script."""
    parser = argparse.ArgumentParser(description="Entropy-Driven CfC Pruning")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick dry-run with minimal data")
    return parser.parse_known_args()[0]
