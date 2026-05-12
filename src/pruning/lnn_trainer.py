"""Training loops for LNN sentence-level (S5a) and token-level (S5b) models."""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

from .lnn_sentence_pruner import LNNSentenceModel
from .lnn_token_pruner import LNNTokenModel
from .lnn_utils import (
    SentenceEncoder,
    MultiTurnSimulator,
    build_sentence_dataset,
    build_token_dataset,
    lnn_sentence_input_dim,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SentenceLNNDataset(Dataset):
    """Each item is one turn's sentence features + labels."""

    def __init__(self, data: dict):
        self.features = data["features"]   # list of (n_i, feat_dim) tensors
        self.labels = data["labels"]       # list of (n_i,) tensors
        self.turns = data["turn_idxs"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.turns[idx]


class TokenLNNDataset(Dataset):
    """Each item is one turn's token features + labels."""

    def __init__(self, data: dict):
        self.features = data["features"]
        self.labels = data["labels"]
        self.turns = data["turn_idxs"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.turns[idx]


def _sentence_pos_weight(labels: list[torch.Tensor], device: str) -> torch.Tensor:
    positives = sum(float((lab > 0.0).sum().item()) for lab in labels)
    total = sum(float(lab.numel()) for lab in labels)
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def _smoothness_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() <= 1:
        return logits.new_tensor(0.0)
    scores = torch.sigmoid(logits)
    return torch.mean((scores[1:] - scores[:-1]) ** 2)


# ---------------------------------------------------------------------------
# Training: Sentence-level (S5a)
# ---------------------------------------------------------------------------

def train_sentence_model(
    train_samples: list[dict],
    val_samples: Optional[list[dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    turns_per_conv: int = 3,
    decay_factor: float = 0.5,
    device: str = None,
    save_path: str = None,
    encoder: Optional[SentenceEncoder] = None,
    early_stopping_patience: int = 3,
    min_delta: float = 1e-4,
    smoothness_weight: float = 0.05,
) -> dict:
    """Train the sentence-level LNN model.

    Args:
        train_samples: list of HotpotQA sample dicts
        val_samples:   optional validation samples
        epochs:        training epochs
        lr:            learning rate
        turns_per_conv: turns per simulated conversation
        decay_factor:  label decay for historical supporting facts
        device:        torch device
        save_path:     path to save model weights
        encoder:       pre-loaded SentenceEncoder (reuse to save memory)

    Returns:
        dict with training history (train_loss, val_loss per epoch)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("[S5a] Building sentence-level training data ...")
    if encoder is None:
        encoder = SentenceEncoder(device=device)
    encoder.load()

    simulator = MultiTurnSimulator(turns_per_conv, decay_factor)
    train_data = build_sentence_dataset(train_samples, encoder, simulator)
    train_ds = SentenceLNNDataset(train_data)

    val_ds = None
    if val_samples:
        val_data = build_sentence_dataset(val_samples, encoder, simulator)
        val_ds = SentenceLNNDataset(val_data)

    emb_dim = encoder.embedding_dim
    input_dim = lnn_sentence_input_dim(emb_dim)

    model = LNNSentenceModel(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_sentence_pos_weight(train_data["labels"], device))

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": None,
        "stopped_early": False,
        "train_val_gap": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"[S5a] Training on {len(train_ds)} conversation-turns, {epochs} epochs ...")

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        # Group by conversation (consecutive turns share hidden state)
        hx = None
        prev_conv_start = None

        for i in range(len(train_ds)):
            features, labels, turn_idx = train_ds[i]
            features = features.to(device)
            labels = labels.to(device)

            # Reset hidden state at conversation start (turn 0)
            if turn_idx == 0:
                hx = None

            logits, hx = model(features, hx)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = labels.clamp(0.0, 1.0)
            loss = criterion(logits, labels)
            loss = loss + smoothness_weight * _smoothness_loss_from_logits(logits)

            # Detach hidden state to avoid backprop through time across turns
            if hx is not None:
                hx = hx.detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        avg_val = 0.0
        if val_ds is not None:
            model.eval()
            val_loss, val_n = 0.0, 0
            hx = None
            with torch.no_grad():
                for i in range(len(val_ds)):
                    feat, lab, ti = val_ds[i]
                    feat, lab = feat.to(device), lab.to(device)
                    if ti == 0:
                        hx = None
                    logits, hx = model(feat, hx)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                    lab = torch.nan_to_num(lab, nan=0.0, posinf=1.0, neginf=0.0)
                    lab = lab.clamp(0.0, 1.0)
                    loss = criterion(logits, lab)
                    loss = loss + smoothness_weight * _smoothness_loss_from_logits(logits)
                    val_loss += loss.item()
                    val_n += 1
            avg_val = val_loss / max(val_n, 1)
            history["val_loss"].append(avg_val)
            history["train_val_gap"].append(avg_val - avg_train)

            if avg_val < best_val_loss - min_delta:
                best_val_loss = avg_val
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = avg_val
                epochs_without_improvement = 0
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improvement += 1

        print(
            f"  Epoch {epoch+1:>3}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )

        if val_ds is not None and epochs_without_improvement >= early_stopping_patience:
            history["stopped_early"] = True
            print(
                f"[S5a] Early stopping at epoch {epoch + 1}; "
                f"best_epoch={history['best_epoch']} best_val_loss={best_val_loss:.4f}"
            )
            break

    # Save final model if no val set was used
    if save_path and val_ds is None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        history["best_epoch"] = len(history["train_loss"])

    print(f"[S5a] Training complete. Model saved to {save_path}")
    return history


# ---------------------------------------------------------------------------
# Training: Token-level (S5b)
# ---------------------------------------------------------------------------

def train_token_model(
    train_samples: list[dict],
    val_samples: Optional[list[dict]] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    turns_per_conv: int = 3,
    decay_factor: float = 0.5,
    device: str = None,
    save_path: str = None,
    encoder: Optional[SentenceEncoder] = None,
) -> dict:
    """Train the token-level LNN model.

    Args & Returns: same structure as train_sentence_model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("[S5b] Building token-level training data ...")
    if encoder is None:
        encoder = SentenceEncoder(device=device)
    encoder.load()

    simulator = MultiTurnSimulator(turns_per_conv, decay_factor)
    train_data = build_token_dataset(train_samples, encoder, simulator)
    train_ds = TokenLNNDataset(train_data)

    val_ds = None
    if val_samples:
        val_data = build_token_dataset(val_samples, encoder, simulator)
        val_ds = TokenLNNDataset(val_data)

    emb_dim = encoder.embedding_dim
    input_dim = emb_dim * 2 + 1

    model = LNNTokenModel(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"[S5b] Training on {len(train_ds)} conversation-turns, {epochs} epochs ...")

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        hx = None

        for i in range(len(train_ds)):
            features, labels, turn_idx = train_ds[i]
            features = features.unsqueeze(0).to(device)  # (1, seq_len, feat_dim)
            labels = labels.to(device)

            if turn_idx == 0:
                hx = None

            scores, hx = model(features, hx)
            # Trim labels to match scores length
            min_len = min(scores.size(0), labels.size(0))
            scores_slice = torch.nan_to_num(scores[:min_len], nan=0.5, posinf=1.0, neginf=0.0)
            scores_slice = scores_slice.clamp(1e-6, 1.0 - 1e-6)
            labels_slice = torch.nan_to_num(labels[:min_len], nan=0.0, posinf=1.0, neginf=0.0)
            labels_slice = labels_slice.clamp(0.0, 1.0)
            loss = criterion(scores_slice, labels_slice)

            if hx is not None:
                hx = hx.detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        avg_val = 0.0
        if val_ds is not None:
            model.eval()
            val_loss, val_n = 0.0, 0
            hx = None
            with torch.no_grad():
                for i in range(len(val_ds)):
                    feat, lab, ti = val_ds[i]
                    feat = feat.unsqueeze(0).to(device)
                    lab = lab.to(device)
                    if ti == 0:
                        hx = None
                    scores, hx = model(feat, hx)
                    ml = min(scores.size(0), lab.size(0))
                    scores_slice = torch.nan_to_num(scores[:ml], nan=0.5, posinf=1.0, neginf=0.0)
                    scores_slice = scores_slice.clamp(1e-6, 1.0 - 1e-6)
                    lab_slice = torch.nan_to_num(lab[:ml], nan=0.0, posinf=1.0, neginf=0.0)
                    lab_slice = lab_slice.clamp(0.0, 1.0)
                    val_loss += criterion(scores_slice, lab_slice).item()
                    val_n += 1
            avg_val = val_loss / max(val_n, 1)
            history["val_loss"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        print(f"  Epoch {epoch+1:>3}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    if save_path and val_ds is None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    print(f"[S5b] Training complete. Model saved to {save_path}")
    return history
