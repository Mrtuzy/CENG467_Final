"""
Faz 4 – CfC Ağının Eğitimi
============================
Sürekli-zamanlı CfC (LTC hücreleri) ağını, öğretmen modelin
ürettiği önem skorlarını tahmin edecek şekilde eğitir.

Kullanım:
    python src/train_cfc.py [--smoke_test]
"""
import os, sys, json, torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (get_args, OUTPUT_DIR, MODEL_DIR, FIGURES_DIR,
                    SBERT_DIM, CFC_UNITS, BATCH_SIZE, LEARNING_RATE, EPOCHS, TAU)

try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
except ImportError:
    raise ImportError("ncps kütüphanesi bulunamadı: pip install ncps")


# ------------------------------------------------------------------ #
#  Dataset & collate                                                  #
# ------------------------------------------------------------------ #
class PruningDataset(Dataset):
    def __init__(self, embeddings, delta_ts, targets):
        # Sadece boş olmayan örnekleri tut
        self.items = [
            (e, d, t) for e, d, t in zip(embeddings, delta_ts, targets)
            if e.size(0) > 0
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    embeddings, delta_ts, targets = zip(*batch)
    seq_lens = [e.size(0) for e in embeddings]
    max_len  = max(seq_lens)
    B = len(batch)
    D = embeddings[0].size(1)

    padded_emb = torch.zeros(B, max_len, D)
    padded_dt  = torch.zeros(B, max_len)          # CfC expects (B, T)
    padded_tgt = torch.zeros(B, max_len)
    mask       = torch.zeros(B, max_len, dtype=torch.bool)

    for i, L in enumerate(seq_lens):
        padded_emb[i, :L] = embeddings[i]
        padded_dt[i, :L]  = delta_ts[i]
        padded_tgt[i, :L] = targets[i]
        mask[i, :L]        = True

    return padded_emb, padded_dt, padded_tgt, mask


# ------------------------------------------------------------------ #
#  Model                                                              #
# ------------------------------------------------------------------ #
class CfCPruner(nn.Module):
    """CfC tabanlı önem-skoru tahmincisi."""
    def __init__(self, input_size: int = 384, hidden_size: int = 64):
        super().__init__()
        wiring = AutoNCP(hidden_size, 1)          # 1 motor nöron
        self.cfc = CfC(input_size, wiring, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, timespans=None):
        # x : (B, T, D)   timespans : (B, T)
        if timespans is not None and x.size(0) > 1:
            # ncps squeezes each time step's timespan internally. With B > 1,
            # that leaves a (B,) tensor that cannot broadcast over hidden units.
            outputs = []
            for i in range(x.size(0)):
                out_i, _ = self.cfc(x[i:i+1], timespans=timespans[i:i+1])
                outputs.append(out_i)
            out = torch.cat(outputs, dim=0)
            return self.sigmoid(out).squeeze(-1)

        out, _ = self.cfc(x, timespans=timespans)
        return self.sigmoid(out).squeeze(-1)   # (B, T)


# ------------------------------------------------------------------ #
#  Training loop                                                      #
# ------------------------------------------------------------------ #
def _eval_epoch(model, loader, criterion, device):
    """Val loss + binary F1 (TAU threshold) üzerinden değerlendirme."""
    model.eval()
    total_loss, total_weight = 0.0, 0.0
    all_pred, all_tgt = [], []

    with torch.no_grad():
        for emb, dt, tgt, mask in loader:
            emb, dt, tgt, mask = (x.to(device) for x in (emb, dt, tgt, mask))
            pred = model(emb, timespans=dt)
            loss = criterion(pred, tgt)
            w = mask.sum().item()
            total_loss   += ((loss * mask).sum().item())
            total_weight += w
            # collect masked predictions/targets for F1
            all_pred.append(pred[mask].cpu().numpy())
            all_tgt.append(tgt[mask].cpu().numpy())

    avg_loss = total_loss / (total_weight + 1e-8)
    preds = np.concatenate(all_pred)
    tgts  = np.concatenate(all_tgt)

    # Binary F1: important turn = score >= TAU
    pred_bin = (preds >= TAU).astype(int)
    tgt_bin  = (tgts  >= TAU).astype(int)
    tp = ((pred_bin == 1) & (tgt_bin == 1)).sum()
    fp = ((pred_bin == 1) & (tgt_bin == 0)).sum()
    fn = ((pred_bin == 0) & (tgt_bin == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    # MAE
    mae = float(np.mean(np.abs(preds - tgts)))

    return avg_loss, float(f1), float(prec), float(rec), mae


def train_cfc(smoke_test: bool = False):
    emb_path = os.path.join(OUTPUT_DIR, "embeddings.pt")
    dt_path  = os.path.join(OUTPUT_DIR, "delta_t.pt")
    tgt_path = os.path.join(OUTPUT_DIR, "teacher_targets.pt")

    for p in (emb_path, dt_path, tgt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} bulunamadı.")

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    delta_ts   = torch.load(dt_path,  map_location="cpu", weights_only=False)
    targets    = torch.load(tgt_path, map_location="cpu", weights_only=False)

    if smoke_test:
        embeddings, delta_ts, targets = embeddings[:10], delta_ts[:10], targets[:10]
        epochs = 3
    else:
        epochs = EPOCHS

    full_dataset = PruningDataset(embeddings, delta_ts, targets)
    if len(full_dataset) == 0:
        print("Uyarı: Eğitim verisi boş, eğitim atlanıyor.")
        return

    # 80 / 20 train-val split
    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CfCPruner(input_size=SBERT_DIM, hidden_size=CFC_UNITS).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_f1": [],
               "val_precision": [], "val_recall": [], "val_mae": []}

    print(f"Eğitim başlıyor – {epochs} epoch | train={train_size} val={val_size}")
    print(f"{'Epoch':>6} {'TrainLoss':>11} {'ValLoss':>9} {'F1':>7} {'Prec':>7} {'Rec':>7} {'MAE':>7}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for emb, dt, tgt, mask in train_loader:
            emb, dt, tgt, mask = (x.to(device) for x in (emb, dt, tgt, mask))
            optimizer.zero_grad()
            pred = model(emb, timespans=dt)
            loss = criterion(pred, tgt)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_loss, f1, prec, rec, mae = _eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)
        history["val_mae"].append(mae)

        print(f"{epoch:>6} {train_loss:>11.6f} {val_loss:>9.6f} {f1:>7.4f} {prec:>7.4f} {rec:>7.4f} {mae:>7.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_cfc_model.pth"))

    # ---- Metrikleri JSON olarak kaydet ----
    metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Metrikler → {metrics_path}")

    # ---- Grafikler ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_x = range(1, epochs + 1)

        # 1. Train vs Val Loss
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, history["train_loss"], marker="o", label="Train Loss")
        ax.plot(epochs_x, history["val_loss"],   marker="s", label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title("CfC Train / Val Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "train_val_loss.png"), dpi=150)
        plt.close()

        # 2. Val F1 / Precision / Recall
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, history["val_f1"],        marker="o", label="F1")
        ax.plot(epochs_x, history["val_precision"], marker="s", label="Precision")
        ax.plot(epochs_x, history["val_recall"],    marker="^", label="Recall")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_title(f"Validation F1 / Precision / Recall  (τ={TAU})")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "val_f1_curve.png"), dpi=150)
        plt.close()

        # 3. Val MAE
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs_x, history["val_mae"], marker="o", color="tab:orange")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MAE")
        ax.set_title("Validation MAE")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "val_mae_curve.png"), dpi=150)
        plt.close()

        print(f"✓ Grafikler → {FIGURES_DIR}")
    except Exception as e:
        print(f"Grafik oluşturulamadı: {e}")

    print(f"✓ En iyi model (val_loss={best_val_loss:.6f}) → {MODEL_DIR}")


if __name__ == "__main__":
    args = get_args()
    train_cfc(smoke_test=args.smoke_test)
