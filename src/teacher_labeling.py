"""
Faz 2 – Öğretmen Model ile Önem Etiketlemesi
=============================================
Hafif bir instruction-tuned LLM (config.TEACHER_MODEL_NAME) ile her
diyalogdaki her turn için **cevap-koşullu leave-one-out ΔNLL** önem skoru
hesaplar:

    önem_i = NLL(cevap | bağlam − turn_i)  −  NLL(cevap | tam bağlam)

Yani bir turn'ü attığımızda gold cevabın olabilirliği ne kadar düşüyorsa
(NLL ne kadar artıyorsa) o turn o kadar önemlidir.  Generation veya tüm
kelime dağılımı (full-vocab KL) gerekmez → eski Mistral-7B KL'ye göre çok
daha hızlı ve "cevap için önem"i doğrudan ölçer.

Kullanım:
    python src/teacher_labeling.py [--smoke_test]
"""
import os, sys, torch
from tqdm import tqdm
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_args, DATA_DIR, OUTPUT_DIR, TEACHER_MODEL_NAME, MAX_TRAIN_SAMPLES
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


def _answer_nll(model, tokenizer, prompt: str, answer: str, device) -> float:
    """Gold cevabın teacher-forcing ortalama NLL'si (sadece cevap tokenları)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=2048).input_ids
    full_ids = tokenizer(prompt + answer, return_tensors="pt",
                         truncation=True, max_length=2048).input_ids.to(device)

    n_prompt = prompt_ids.size(1)
    if full_ids.size(1) <= n_prompt:          # cevap kırpıldı / boş
        return 0.0

    labels = full_ids.clone()
    labels[:, :n_prompt] = -100               # prompt tokenlarını maskele
    with torch.no_grad():
        loss = model(full_ids, labels=labels).loss
    return float(loss.item())


def generate_teacher_labels(smoke_test: bool = False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} bulunamadı. Önce data_prep.py çalıştırın.")

    train_ds = load_from_disk(train_path)
    if smoke_test:
        n = min(5, len(train_ds))
        print(f"[SMOKE TEST] Sadece {n} örnek işlenecek.")
        train_ds = train_ds.select(range(n))
    else:
        n = min(MAX_TRAIN_SAMPLES, len(train_ds))
        if n < len(train_ds):
            print(f"[Limit] {len(train_ds)} örnekten {n} tanesi kullanılacak (MAX_TRAIN_SAMPLES).")
        train_ds = train_ds.select(range(n))

    # ---- model yükleme ----
    # Qwen2.5-1.5B gated değil ve küçük → quantization gerekmez, fp16 yeterli.
    print(f"Öğretmen model yükleniyor: {TEACHER_MODEL_NAME}")
    tokenizer = load_tokenizer(TEACHER_MODEL_NAME)
    model = load_causal_lm(
        TEACHER_MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
        quantize_4bit=False,
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

    print("Cevap-koşullu leave-one-out ΔNLL etiketleme başlatılıyor …")
    for idx, example in enumerate(tqdm(train_ds)):
        if idx < start_idx:
            continue

        context  = example["context"]
        question = example["question"]
        answer   = example.get("answer", "") or ""

        if not context or not answer.strip():
            all_targets.append(torch.tensor([], dtype=torch.float32))
        else:
            # Tam bağlamla cevabın NLL'si
            full_prompt = _build_prompt(tokenizer, context, question)
            nll_full = _answer_nll(model, tokenizer, full_prompt, answer, input_device)

            # Her turn'ü teker teker çıkarıp cevap NLL'sini ölç → ΔNLL
            scores = []
            for i in range(len(context)):
                abl_prompt = _build_prompt(
                    tokenizer, context[:i] + context[i+1:], question)
                nll_abl = _answer_nll(model, tokenizer, abl_prompt, answer, input_device)
                scores.append(max(0.0, nll_abl - nll_full))   # önem = NLL artışı

            # [0, 1] normalize  (CfC hedef ölçeği değişmesin)
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
