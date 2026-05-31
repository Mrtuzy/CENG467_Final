import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ncps.torch import CfC
from config import get_args, OUTPUT_DIR, MODEL_DIR, CFC_UNITS, BATCH_SIZE, LEARNING_RATE, EPOCHS

class PruningDataset(Dataset):
    def __init__(self, embeddings, delta_ts, targets):
        self.embeddings = embeddings
        self.delta_ts = delta_ts
        self.targets = targets
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.delta_ts[idx], self.targets[idx]

def collate_fn(batch):
    embeddings, delta_ts, targets = zip(*batch)
    
    seq_lens = [e.size(0) for e in embeddings]
    if not seq_lens:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
    max_len = max(seq_lens)
    if max_len == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    padded_emb = torch.zeros(len(batch), max_len, embeddings[0].size(1))
    padded_dt = torch.zeros(len(batch), max_len)
    padded_tgt = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i in range(len(batch)):
        l = seq_lens[i]
        if l > 0:
            padded_emb[i, :l, :] = embeddings[i]
            padded_dt[i, :l] = delta_ts[i]
            padded_tgt[i, :l] = targets[i]
            mask[i, :l] = True
            
    return padded_emb, padded_dt, padded_tgt, mask

class CfCPruner(nn.Module):
    def __init__(self, input_size=384, hidden_size=64):
        super().__init__()
        self.cfc = CfC(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, timespans=None):
        out, _ = self.cfc(x, timespans=timespans)
        scores = self.sigmoid(self.fc(out)).squeeze(-1) 
        return scores

def train_cfc(smoke_test=False):
    emb_path = os.path.join(OUTPUT_DIR, "embeddings.pt")
    dt_path = os.path.join(OUTPUT_DIR, "delta_t.pt")
    tgt_path = os.path.join(OUTPUT_DIR, "teacher_targets.pt")
    
    if not (os.path.exists(emb_path) and os.path.exists(dt_path) and os.path.exists(tgt_path)):
        raise FileNotFoundError("Input files for training not found. Run build_inputs.py first.")
        
    embeddings = torch.load(emb_path, map_location="cpu")
    delta_ts = torch.load(dt_path, map_location="cpu")
    targets = torch.load(tgt_path, map_location="cpu")
    
    if smoke_test:
        embeddings = embeddings[:10]
        delta_ts = delta_ts[:10]
        targets = targets[:10]
        epochs = 2
    else:
        epochs = EPOCHS
        
    dataset = PruningDataset(embeddings, delta_ts, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfCPruner(input_size=384, hidden_size=CFC_UNITS).to(device)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for emb, dt, tgt, mask in dataloader:
            if emb.size(0) == 0:
                continue
                
            emb, dt, tgt, mask = emb.to(device), dt.to(device), tgt.to(device), mask.to(device)
            
            optimizer.zero_grad()
            scores = model(emb, timespans=dt)
            
            loss = criterion(scores, tgt)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss and len(dataloader) > 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_cfc_model.pth"))
            
    print(f"Training complete. Best model saved to {MODEL_DIR}")

if __name__ == "__main__":
    args = get_args()
    train_cfc(smoke_test=args.smoke_test)
