"""
Faz 3 – Girdi Vektörizasyonu ve Entropi Hesaplama
==================================================
SBERT ile cümle embeddingleri (e_i) ve DistilGPT-2 ile surprisal
tabanlı zaman adımları (Δt_i) hesaplanır. CfC ağının eğitimi için
hazır tensörler Google Drive'a kaydedilir.

Kullanım:
    python src/build_inputs.py [--smoke_test]
"""
import os, sys, torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (get_args, DATA_DIR, OUTPUT_DIR,
                    PROXY_MODEL_NAME, SBERT_MODEL_NAME,
                    DELTA_T_MIN, BETA, DT_MAPPING, MAX_TRAIN_SAMPLES)


def surprisal_to_dt(surprisals: list[float]) -> list[float]:
    """Bir bağlamın surprisal'larını Δt'ye çevir (config.DT_MAPPING'e göre).

    - "linear": Δt = Δt_min + β·S                (sınırsız, doğrusal)
    - "rank"  : Δt = Δt_min + β·rank(S)/N         (sıra-normalize; aykırı surprisal'a
                dayanıklı — bkz. simulations/cfc_proof S5). Rank, bağlam İÇİNDE alınır.
    """
    n = len(surprisals)
    if DT_MAPPING == "rank":
        if n <= 1:
            return [DELTA_T_MIN + BETA * 0.5 for _ in surprisals]
        # ortalama-sıra (bağ durumunda), (0,1] aralığına normalize
        order = sorted(range(n), key=lambda i: surprisals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and surprisals[order[j + 1]] == surprisals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-tabanlı ortalama sıra
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank / n
            i = j + 1
        return [DELTA_T_MIN + BETA * r for r in ranks]
    # varsayılan: linear
    return [DELTA_T_MIN + BETA * s for s in surprisals]


def calculate_surprisal(text: str, model, tokenizer, device: str) -> float:
    """Ortalama negatif log-olasılık (surprisal) hesapla."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    if inputs.input_ids.size(1) < 2:
        return 0.0
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return loss.item()


def calculate_surprisal_batch(texts: list[str], model, tokenizer, device: str) -> list[float]:
    """Birden fazla metni tek forward pass ile surprisal hesapla."""
    if not texts:
        return []
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        # per-token loss için logits'i manuel hesapla
        logits = out.logits  # (B, T, V)
    import torch.nn.functional as F
    B, T, V = logits.shape
    # shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels = enc["input_ids"][:, 1:].contiguous().view(-1)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    losses = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_id, reduction="none")
    losses = losses.view(B, T - 1)
    attn = enc["attention_mask"][:, 1:].float()
    per_sample = (losses * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
    return per_sample.cpu().tolist()


def build_inputs(smoke_test: bool = False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} bulunamadı.")

    train_ds = load_from_disk(train_path)
    if smoke_test:
        train_ds = train_ds.select(range(min(5, len(train_ds))))
    else:
        n = min(MAX_TRAIN_SAMPLES, len(train_ds))
        if n < len(train_ds):
            print(f"[Limit] {len(train_ds)} örnekten {n} tanesi kullanılacak (MAX_TRAIN_SAMPLES).")
        train_ds = train_ds.select(range(n))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- modelleri yükle ----
    print(f"Proxy model (entropi): {PROXY_MODEL_NAME}")
    proxy_tok = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    if proxy_tok.pad_token is None:          # distilgpt2'nin pad token'ı yok
        proxy_tok.pad_token = proxy_tok.eos_token
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()

    print(f"SBERT (vektörizasyon): {SBERT_MODEL_NAME}")
    sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_emb, all_dt = [], []

    SURPRISAL_BATCH = 32  # DistilGPT2 küçük, daha büyük batch GPU'da hızlı

    print("Embedding + Δt hesaplanıyor …")
    for example in tqdm(train_ds):
        ctx = example["context"]
        if not ctx:
            all_emb.append(torch.empty((0, 384)))
            all_dt.append(torch.empty(0))
            continue

        # SBERT embedding (zaten batch)
        emb = sbert.encode(ctx, convert_to_tensor=True,
                           show_progress_bar=False)
        all_emb.append(emb.cpu())

        # Δt = f(S(u))  — batch surprisal, sonra config.DT_MAPPING eşlemesi
        surprisals = []
        for i in range(0, len(ctx), SURPRISAL_BATCH):
            chunk = ctx[i : i + SURPRISAL_BATCH]
            surprisals.extend(calculate_surprisal_batch(chunk, proxy_model, proxy_tok, device))
        dts = surprisal_to_dt(surprisals)
        all_dt.append(torch.tensor(dts, dtype=torch.float32))

    torch.save(all_emb, os.path.join(OUTPUT_DIR, "embeddings.pt"))
    torch.save(all_dt,  os.path.join(OUTPUT_DIR, "delta_t.pt"))
    print(f"✓ Kaydedildi → {OUTPUT_DIR}")


if __name__ == "__main__":
    args = get_args()
    build_inputs(smoke_test=args.smoke_test)
