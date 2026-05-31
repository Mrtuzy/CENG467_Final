"""
Faz 4 – CfC Ağının Eğitimi
============================
Sürekli-zamanlı CfC (LTC hücreleri) ağını, öğretmen modelin
ürettiği önem skorlarını tahmin edecek şekilde eğitir.

Kullanım:
    python src/train_cfc.py [--smoke_test]
"""
import os, sys, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (get_args, OUTPUT_DIR, MODEL_DIR, FIGURES_DIR,
                    SBERT_DIM, CFC_UNITS, BATCH_SIZE, LEARNING_RATE, EPOCHS)

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
        out, _ = self.cfc(x, timespans=timespans)
        return self.sigmoid(out).squeeze(-1)   # (B, T)


# ------------------------------------------------------------------ #
#  Training loop                                                      #
# ------------------------------------------------------------------ #
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

    dataset    = PruningDataset(embeddings, delta_ts, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn)
    if len(dataset) == 0:
        print("Uyarı: Eğitim verisi boş, eğitim atlanıyor.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CfCPruner(input_size=SBERT_DIM, hidden_size=CFC_UNITS).to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    best_loss = float("inf")
    history = []

    print(f"Eğitim başlıyor – {epochs} epoch, {len(dataset)} örnek …")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for emb, dt, tgt, mask in dataloader:
            emb  = emb.to(device)
            dt   = dt.to(device)
            tgt  = tgt.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(emb, timespans=dt)
            loss = criterion(pred, tgt)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg = running / len(dataloader)
        history.append(avg)
        print(f"  Epoch {epoch}/{epochs}  loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_cfc_model.pth"))

    # ---- Training loss grafiği kaydet ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history)+1), history, marker="o", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("CfC Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "training_loss.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Training loss grafiği → {fig_path}")
    except Exception as e:
        print(f"Grafik oluşturulamadı: {e}")

    print(f"✓ En iyi model → {MODEL_DIR}")


if __name__ == "__main__":
    args = get_args()
    train_cfc(smoke_test=args.smoke_test)
