import argparse
import os

# Google Drive Paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/CENG_467"
DATA_DIR = os.path.join(DRIVE_BASE_DIR, "data")
MODEL_DIR = os.path.join(DRIVE_BASE_DIR, "models")
OUTPUT_DIR = os.path.join(DRIVE_BASE_DIR, "outputs")

# Proxy & Teacher Models
PROXY_MODEL_NAME = "distilgpt2"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
TEACHER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# CfC Hyperparameters
DELTA_T_MIN = 0.1
BETA = 1.0
CFC_UNITS = 64
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10

def get_args():
    parser = argparse.ArgumentParser(description="Entropy-Driven CfC Pruning")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick test with very little data")
    return parser.parse_known_args()[0]
