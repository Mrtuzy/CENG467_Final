"""
Faz 5 – Uçtan Uca Değerlendirme ve Analiz
==========================================
Eğitilen CfC modeli ile test seti üzerinde çıkarım yapar.
Baseline (Full, Random, Cosine) ile karşılaştırıp kalite
(ROUGE-L, BERTScore) ve hız (TTFT) metriklerini hesaplar.
Sonuçları tablo ve grafik olarak Google Drive'a kaydeder.

Kullanım:
    python src/evaluate.py [--smoke_test]
"""
import os, sys, time, json
import torch
import numpy as np

RANDOM_SEED = 42
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
                    SBERT_DIM, CFC_UNITS, DELTA_T_MIN, BETA, TAU)
from train_cfc import CfCPruner
from build_inputs import calculate_surprisal
from model_loading import load_causal_lm, load_tokenizer, model_input_device


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def _build_user_content(context_list, question):
    p = ""
    if context_list:
        p += "Conversation history:\n"
        for i, t in enumerate(context_list):
            p += f"Turn {i+1}: {t}\n"
        p += "\n"
    p += f"Question: {question}"
    return p


def _build_prompt(tokenizer, context_list, question):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer concisely based on the conversation history.",
        },
        {"role": "user", "content": _build_user_content(context_list, question)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {messages[0]['content']}\n\n{messages[1]['content']} [/INST]"


def _generate(model, tokenizer, prompt, max_new=50):
    input_device = model_input_device(model)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=2048).to(input_device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=False)
    ttft = time.time() - t0
    text = tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()
    return text, ttft


def _count_tokens(tokenizer, texts):
    return sum(len(tokenizer.encode(t)) for t in texts)


# ------------------------------------------------------------------ #
#  Main evaluation                                                    #
# ------------------------------------------------------------------ #
def run_evaluation(smoke_test: bool = False):
    test_path = os.path.join(DATA_DIR, "test_processed")
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test verisi bulunamadı.")

    test_ds = load_from_disk(test_path)
    if smoke_test:
        test_ds = test_ds.select(range(min(5, len(test_ds))))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- lightweight modeller ----
    print("Modeller yükleniyor …")
    proxy_tok   = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()

    sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    cfc = CfCPruner(input_size=SBERT_DIM, hidden_size=CFC_UNITS).to(device)
    wpath = os.path.join(MODEL_DIR, "best_cfc_model.pth")
    if os.path.exists(wpath):
        cfc.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
    else:
        print("⚠  Eğitilmiş CfC ağırlıkları yok – rastgele ağırlıklar kullanılıyor.")
    cfc.eval()

    # ---- LLM (teacher) ----
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
    methods = ["full", "cfc", "random", "cosine"]
    R = {m: {"rougeL": [], "ttft": [], "tokens": []} for m in methods}
    all_refs = []
    all_preds = {m: [] for m in methods}

    rng = np.random.default_rng(RANDOM_SEED)

    print("Değerlendirme başlıyor …")
    for example in tqdm(test_ds):
        ctx = example["context"]
        q   = example["question"]
        gt  = example["answer"]
        if not ctx:
            continue

        all_refs.append(gt)
        full_tokens = _count_tokens(llm_tok, ctx)

        # --- (1) Full Context ---
        ans, ttft = _generate(llm, llm_tok, _build_prompt(llm_tok, ctx, q))
        R["full"]["rougeL"].append(rouge.score(gt, ans)["rougeL"].fmeasure)
        R["full"]["ttft"].append(ttft)
        R["full"]["tokens"].append(full_tokens)
        all_preds["full"].append(ans)

        # --- (2) CfC Pruning ---
        embs = sbert.encode(ctx, convert_to_tensor=True, show_progress_bar=False)
        dts = [DELTA_T_MIN + BETA * calculate_surprisal(u, proxy_model, proxy_tok, device)
               for u in ctx]
        embs_b = embs.unsqueeze(0).to(device)
        dts_b  = torch.tensor(dts, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = cfc(embs_b, timespans=dts_b).squeeze(0).cpu().numpy()
        cfc_ctx = [u for u, s in zip(ctx, scores) if s >= TAU]
        if not cfc_ctx:  # en azından en yüksek skorlu 1 cümleyi tut
            best_idx = int(np.argmax(scores))
            cfc_ctx = [ctx[best_idx]]
        ans, ttft = _generate(llm, llm_tok, _build_prompt(llm_tok, cfc_ctx, q))
        R["cfc"]["rougeL"].append(rouge.score(gt, ans)["rougeL"].fmeasure)
        R["cfc"]["ttft"].append(ttft)
        R["cfc"]["tokens"].append(_count_tokens(llm_tok, cfc_ctx))
        all_preds["cfc"].append(ans)
        keep = len(cfc_ctx)

        # --- (3) Random Pruning ---
        if keep < len(ctx):
            idx = np.sort(rng.choice(len(ctx), keep, replace=False))
            rand_ctx = [ctx[i] for i in idx]
        else:
            rand_ctx = ctx
        ans, ttft = _generate(llm, llm_tok, _build_prompt(llm_tok, rand_ctx, q))
        R["random"]["rougeL"].append(rouge.score(gt, ans)["rougeL"].fmeasure)
        R["random"]["ttft"].append(ttft)
        R["random"]["tokens"].append(_count_tokens(llm_tok, rand_ctx))
        all_preds["random"].append(ans)

        # --- (4) Cosine Similarity Pruning ---
        q_emb = sbert.encode(q, convert_to_tensor=True)
        cos = util.cos_sim(q_emb, embs)[0].cpu().numpy()
        if keep > 0:
            top = np.sort(np.argsort(cos)[-keep:])
            cos_ctx = [ctx[i] for i in top]
        else:
            cos_ctx = []
        ans, ttft = _generate(llm, llm_tok, _build_prompt(llm_tok, cos_ctx, q))
        R["cosine"]["rougeL"].append(rouge.score(gt, ans)["rougeL"].fmeasure)
        R["cosine"]["ttft"].append(ttft)
        R["cosine"]["tokens"].append(_count_tokens(llm_tok, cos_ctx))
        all_preds["cosine"].append(ans)

    # ================================================================ #
    #  Sonuçları topla                                                  #
    # ================================================================ #
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}
    full_tok_avg = np.mean(R["full"]["tokens"]) if R["full"]["tokens"] else 1
    print("\n" + "="*65)
    print(f'{"Method":<12} {"ROUGE-L":>10} {"TTFT (s)":>10} {"Tokens":>10} {"Reduction%":>12}')
    print("-"*65)
    for m in methods:
        rl   = np.mean(R[m]["rougeL"]) if R[m]["rougeL"] else 0
        ttft = np.mean(R[m]["ttft"])   if R[m]["ttft"]   else 0
        tok  = np.mean(R[m]["tokens"]) if R[m]["tokens"] else 0
        red  = (1 - tok / full_tok_avg) * 100 if full_tok_avg else 0
        summary[m] = {"ROUGE-L": round(rl, 4), "TTFT": round(ttft, 4),
                      "Avg_Tokens": round(tok, 1), "Reduction%": round(red, 1)}
        print(f"{m:<12} {rl:>10.4f} {ttft:>10.4f} {tok:>10.1f} {red:>11.1f}%")
    print("="*65)

    # ---- BERTScore (sadece gerçek koşumda) ----
    if not smoke_test and all_refs:
        try:
            import evaluate as hf_evaluate
            bertscore = hf_evaluate.load("bertscore")
            print("\nBERTScore hesaplanıyor …")
            for m in methods:
                bs = bertscore.compute(predictions=all_preds[m],
                                       references=all_refs, lang="en")
                f1 = round(np.mean(bs["f1"]), 4)
                summary[m]["BERTScore_F1"] = f1
                print(f"  [{m.upper()}] BERTScore F1: {f1}")
        except Exception as e:
            print(f"  BERTScore atlandı: {e}")

    # ---- Kaydet ----
    json_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Sonuçlar → {json_path}")

    # ================================================================ #
    #  Grafikler                                                       #
    # ================================================================ #
    labels = [m.upper() for m in methods]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    def _bar(values, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(values)*0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  📊 {path}")

    print("\nGrafikler oluşturuluyor …")
    _bar([summary[m]["ROUGE-L"] for m in methods],
         "ROUGE-L", "ROUGE-L Comparison", "rouge_l_comparison.png")
    _bar([summary[m]["TTFT"] for m in methods],
         "Seconds", "Time To First Token (TTFT)", "ttft_comparison.png")
    _bar([summary[m]["Avg_Tokens"] for m in methods],
         "Tokens", "Average Context Token Count", "token_count_comparison.png")
    _bar([summary[m]["Reduction%"] for m in methods],
         "% Reduction", "Token Reduction vs Full Context", "token_reduction_comparison.png")

    if "BERTScore_F1" in summary.get("full", {}):
        _bar([summary[m]["BERTScore_F1"] for m in methods],
             "F1", "BERTScore F1 Comparison", "bertscore_comparison.png")

    print("\n✓ Değerlendirme tamamlandı.")


if __name__ == "__main__":
    args = get_args()
    run_evaluation(smoke_test=args.smoke_test)
