"""
Ablation Study – TAU Eşiği Sweep
=================================
Eğitilmiş CfC modelini farklı pruning eşikleri (TAU) ile çalıştırarak
ROUGE-L ve token reduction'ın TAU'ya duyarlılığını ölçer.

Kullanım:
    python src/ablation.py [--smoke_test]
"""
import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (get_args, DATA_DIR, OUTPUT_DIR, MODEL_DIR, FIGURES_DIR,
                    TEACHER_MODEL_NAME, PROXY_MODEL_NAME, SBERT_MODEL_NAME,
                    SBERT_DIM, CFC_UNITS, DELTA_T_MIN, BETA, ABLATION_MAX_SAMPLES)
from train_cfc import CfCPruner
from build_inputs import calculate_surprisal, surprisal_to_dt
from model_loading import load_causal_lm, load_tokenizer, model_input_device
from evaluate import _build_prompt, _generate, _count_tokens

# 5 nokta, kalite/verimlilik eğrisini göstermeye yeter (eski 7'den hızlı)
TAU_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_ablation(smoke_test: bool = False):
    test_path = os.path.join(DATA_DIR, "test_processed")
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test verisi bulunamadı.")

    test_ds = load_from_disk(test_path)
    if smoke_test:
        test_ds = test_ds.select(range(min(5, len(test_ds))))
    else:
        n = min(ABLATION_MAX_SAMPLES, len(test_ds))
        if n < len(test_ds):
            print(f"[Limit] Ablation {len(test_ds)} → {n} örnek (ABLATION_MAX_SAMPLES).")
        test_ds = test_ds.select(range(n))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Modeller yükleniyor …")
    proxy_tok   = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()

    sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    cfc = CfCPruner(input_size=SBERT_DIM, hidden_size=CFC_UNITS).to(device)
    wpath = os.path.join(MODEL_DIR, "best_cfc_model.pth")
    if os.path.exists(wpath):
        cfc.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
        print("✓ CfC ağırlıkları yüklendi.")
    else:
        print("⚠  Eğitilmiş ağırlık bulunamadı – rastgele ağırlıklar kullanılıyor.")
    cfc.eval()

    print(f"LLM yükleniyor: {TEACHER_MODEL_NAME}")
    llm_tok = load_tokenizer(TEACHER_MODEL_NAME)
    llm = load_causal_lm(
        TEACHER_MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
        quantize_4bit=True,
    )
    llm.eval()

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # --- Önce embeddings & scores'ları hesapla (model başına bir kez) ---
    print("\nEmbedding ve CfC skorları hesaplanıyor …")
    cached = []
    for example in tqdm(test_ds):
        ctx = example["context"]
        q   = example["question"]
        gt  = example["answer"]
        if not ctx:
            continue

        embs = sbert.encode(ctx, convert_to_tensor=True, show_progress_bar=False)
        # Δt: eğitimle AYNI eşleme (config.DT_MAPPING, surprisal_to_dt) kullanılır.
        surprisals = [calculate_surprisal(u, proxy_model, proxy_tok, device) for u in ctx]
        dts  = surprisal_to_dt(surprisals)
        embs_b = embs.unsqueeze(0).to(device)
        dts_b  = torch.tensor(dts, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = cfc(embs_b, timespans=dts_b).squeeze(0).cpu().numpy()

        full_tokens = _count_tokens(llm_tok, ctx)
        cached.append({"ctx": ctx, "q": q, "gt": gt,
                       "scores": scores, "full_tokens": full_tokens})

    # --- TAU sweep ---
    results = {}
    for tau in TAU_VALUES:
        print(f"\n── TAU = {tau} ──")
        rouges, token_counts = [], []
        for item in tqdm(cached):
            ctx, q, gt, scores = item["ctx"], item["q"], item["gt"], item["scores"]
            pruned_ctx = [u for u, s in zip(ctx, scores) if s >= tau]
            if not pruned_ctx:
                pruned_ctx = [ctx[int(np.argmax(np.nan_to_num(scores)))]]

            ans, _ = _generate(llm, llm_tok, _build_prompt(llm_tok, pruned_ctx, q))
            rl = rouge.score(gt, ans)["rougeL"].fmeasure
            rouges.append(rl)
            token_counts.append(_count_tokens(llm_tok, pruned_ctx))

        full_avg = np.mean([it["full_tokens"] for it in cached]) or 1
        avg_rl   = round(float(np.mean(rouges)), 4)
        avg_red  = round((1 - np.mean(token_counts) / full_avg) * 100, 1)
        avg_kept = round(float(np.mean(token_counts)), 1)

        results[tau] = {
            "ROUGE-L":    avg_rl,
            "Avg_Tokens": avg_kept,
            "Reduction%": avg_red,
        }
        print(f"  ROUGE-L={avg_rl:.4f}  Tokens={avg_kept:.1f}  Reduction={avg_red:.1f}%")

    # ---- Kaydet ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n✓ Ablasyon sonuçları → {json_path}")

    # ---- Grafik ----
    taus   = TAU_VALUES
    rouges = [results[t]["ROUGE-L"]    for t in taus]
    reds   = [results[t]["Reduction%"] for t in taus]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "#4C72B0"
    ax1.plot(taus, rouges, marker="o", color=color1, linewidth=2, label="ROUGE-L")
    ax1.set_xlabel("TAU (pruning threshold)")
    ax1.set_ylabel("ROUGE-L", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#DD8452"
    ax2.plot(taus, reds, marker="s", color=color2, linewidth=2,
             linestyle="--", label="Reduction%")
    ax2.set_ylabel("Token Reduction %", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plt.title("Ablation: TAU Threshold vs ROUGE-L & Token Reduction")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "ablation_tau_sweep.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Ablasyon grafiği → {fig_path}")


if __name__ == "__main__":
    args = get_args()
    run_ablation(smoke_test=args.smoke_test)
