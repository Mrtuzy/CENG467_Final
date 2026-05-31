"""
Faz 2 – Öğretmen Model ile Altın Standart Etiketleme
=====================================================
LLaMA-3.1-8B-Instruct'ı 4-bit quantization ile yükler.
Her diyalogdaki her cümle için leave-one-out KL-Divergence
yaklaşımı ile önem skoru (p_target) hesaplar ve tensör olarak
Google Drive'a kaydeder.

Kullanım:
    python src/teacher_labeling.py [--smoke_test]
"""
import os, sys, torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_args, DATA_DIR, OUTPUT_DIR, TEACHER_MODEL_NAME


def _build_prompt(context_list, question):
    """Build a LLaMA 3.1 chat prompt."""
    p  = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    p += "You are a helpful AI assistant.<|eot_id|>"
    p += "<|start_header_id|>user<|end_header_id|>\n\n"
    if context_list:
        p += "Conversation history:\n"
        for i, turn in enumerate(context_list):
            p += f"Turn {i+1}: {turn}\n"
        p += "\n"
    p += f"Question: {question}<|eot_id|>"
    p += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return p


def _kl_div(logits_base, logits_ablated):
    """KL-divergence between last-token distributions."""
    p = F.softmax(logits_base[:, -1, :], dim=-1)
    log_q = F.log_softmax(logits_ablated[:, -1, :], dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean").item()


def generate_teacher_labels(smoke_test: bool = False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} bulunamadı. Önce data_prep.py çalıştırın.")

    train_ds = load_from_disk(train_path)
    if smoke_test:
        n = min(5, len(train_ds))
        print(f"[SMOKE TEST] Sadece {n} örnek işlenecek.")
        train_ds = train_ds.select(range(n))

    # ---- model yükleme ----
    print(f"Öğretmen model yükleniyor: {TEACHER_MODEL_NAME}")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Uyari: HF_TOKEN/HUGGINGFACE_TOKEN bulunamadi. Gated modele erisim icin token gerekir.")
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL_NAME,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_targets = []

    print("Leave-one-out etiketleme başlatılıyor …")
    for example in tqdm(train_ds):
        context  = example["context"]
        question = example["question"]

        if not context:
            all_targets.append(torch.tensor([], dtype=torch.float32))
            continue

        # tam bağlam logits
        base_ids = tokenizer(_build_prompt(context, question),
                             return_tensors="pt", truncation=True,
                             max_length=2048).to(model.device)
        with torch.no_grad():
            logits_base = model(**base_ids).logits

        scores = []
        for i in range(len(context)):
            ablated = context[:i] + context[i+1:]
            abl_ids = tokenizer(_build_prompt(ablated, question),
                                return_tensors="pt", truncation=True,
                                max_length=2048).to(model.device)
            with torch.no_grad():
                logits_abl = model(**abl_ids).logits
            scores.append(_kl_div(logits_base, logits_abl))

        # [0, 1] normalize
        mn, mx = min(scores), max(scores)
        if mx > mn:
            normed = [(s - mn) / (mx - mn) for s in scores]
        elif len(scores) == 1:
            normed = [1.0]
        else:
            normed = [0.5] * len(scores)

        all_targets.append(torch.tensor(normed, dtype=torch.float32))

    out_path = os.path.join(OUTPUT_DIR, "teacher_targets.pt")
    torch.save(all_targets, out_path)
    print(f"✓ Etiketler kaydedildi → {out_path}")


if __name__ == "__main__":
    args = get_args()
    generate_teacher_labels(smoke_test=args.smoke_test)
