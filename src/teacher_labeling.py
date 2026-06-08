"""
Faz 2 – Öğretmen Model ile Altın Standart Etiketleme
=====================================================
Mistral-7B-Instruct-v0.2'yi 4-bit quantization ile yükler.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_args, DATA_DIR, OUTPUT_DIR, TEACHER_MODEL_NAME
from model_loading import get_hf_token, load_causal_lm, load_tokenizer, model_input_device


def _build_messages(context_list, question):
    """Build chat messages for an instruction-tuned teacher model."""
    user_parts = ["You are a helpful AI assistant."]
    if context_list:
        history = "\n".join(f"Turn {i+1}: {turn}" for i, turn in enumerate(context_list))
        user_parts.append(f"Conversation history:\n{history}")
    user_parts.append(f"Question: {question}")
    return [{"role": "user", "content": "\n\n".join(user_parts)}]


def _build_prompt(tokenizer, context_list, question):
    messages = _build_messages(context_list, question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {messages[0]['content']} [/INST]"


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
    if not get_hf_token():
        print("Uyari: HF_TOKEN/HUGGINGFACE_TOKEN bulunamadi. Gated modele erisim icin token gerekir.")
    tokenizer = load_tokenizer(
        TEACHER_MODEL_NAME,
    )
    model = load_causal_lm(
        TEACHER_MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
        quantize_4bit=True,
    )
    model.eval()
    input_device = model_input_device(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path   = os.path.join(OUTPUT_DIR, "teacher_targets.pt")
    ckpt_path  = out_path + ".checkpoint"

    # Resume from checkpoint if it exists
    if os.path.exists(ckpt_path):
        all_targets = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        start_idx   = len(all_targets)
        print(f"[Checkpoint] {start_idx} örnek yüklendi, {start_idx}. örnekten devam ediliyor.")
    else:
        all_targets = []
        start_idx   = 0

    print("Leave-one-out etiketleme başlatılıyor …")
    for idx, example in enumerate(tqdm(train_ds)):
        if idx < start_idx:
            continue

        context  = example["context"]
        question = example["question"]

        if not context:
            all_targets.append(torch.tensor([], dtype=torch.float32))
        else:
            # tam bağlam logits
            base_ids = tokenizer(_build_prompt(tokenizer, context, question),
                                 return_tensors="pt", truncation=True,
                                 max_length=2048).to(input_device)
            with torch.no_grad():
                logits_base = model(**base_ids).logits

            scores = []
            for i in range(len(context)):
                ablated = context[:i] + context[i+1:]
                abl_ids = tokenizer(_build_prompt(tokenizer, ablated, question),
                                    return_tensors="pt", truncation=True,
                                    max_length=2048).to(input_device)
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

        # Checkpoint her 100 örnekte bir
        if (len(all_targets) % 100) == 0:
            torch.save(all_targets, ckpt_path)

    torch.save(all_targets, out_path)
    # Tamamlandıktan sonra checkpoint'i temizle
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"✓ Etiketler kaydedildi → {out_path}")


if __name__ == "__main__":
    args = get_args()
    generate_teacher_labels(smoke_test=args.smoke_test)
