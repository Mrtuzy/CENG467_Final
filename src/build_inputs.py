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
                    DELTA_T_MIN, BETA)


def calculate_surprisal(text: str, model, tokenizer, device: str) -> float:
    """Ortalama negatif log-olasılık (surprisal) hesapla."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    if inputs.input_ids.size(1) < 2:
        return 0.0
    with torch.no_grad():
        loss = model(**inputs, labels=inputs.input_ids).loss
    return loss.item()


def build_inputs(smoke_test: bool = False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} bulunamadı.")

    train_ds = load_from_disk(train_path)
    if smoke_test:
        train_ds = train_ds.select(range(min(5, len(train_ds))))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- modelleri yükle ----
    print(f"Proxy model (entropi): {PROXY_MODEL_NAME}")
    proxy_tok = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()

    print(f"SBERT (vektörizasyon): {SBERT_MODEL_NAME}")
    sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_emb, all_dt = [], []

    print("Embedding + Δt hesaplanıyor …")
    for example in tqdm(train_ds):
        ctx = example["context"]
        if not ctx:
            all_emb.append(torch.empty((0, 384)))
            all_dt.append(torch.empty(0))
            continue

        # SBERT embedding
        emb = sbert.encode(ctx, convert_to_tensor=True,
                           show_progress_bar=False)
        all_emb.append(emb.cpu())

        # Δt = Δt_min + β · S(u)
        dts = []
        for utt in ctx:
            s = calculate_surprisal(utt, proxy_model, proxy_tok, device)
            dts.append(DELTA_T_MIN + BETA * s)
        all_dt.append(torch.tensor(dts, dtype=torch.float32))

    torch.save(all_emb, os.path.join(OUTPUT_DIR, "embeddings.pt"))
    torch.save(all_dt,  os.path.join(OUTPUT_DIR, "delta_t.pt"))
    print(f"✓ Kaydedildi → {OUTPUT_DIR}")


if __name__ == "__main__":
    args = get_args()
    build_inputs(smoke_test=args.smoke_test)
