"""
Faz 1 – Veri Seti Hazırlığı
============================
QReCC veri setini ham JSON olarak indirir, diyalog formatına çevirir,
çok kısa / çok uzun diyalogları temizler ve HuggingFace Dataset olarak
Google Drive'a kaydeder.

Kullanım:
    python src/data_prep.py [--smoke_test]
"""
import os, sys, json, requests

# ---- path bootstrap: config.py aynı dizinde ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_args, DATA_DIR
from datasets import Dataset

TRAIN_URL = "https://huggingface.co/datasets/svakulenk0/qrecc/resolve/main/qrecc-training.json"
TEST_URL  = "https://huggingface.co/datasets/svakulenk0/qrecc/resolve/main/qrecc-test.json"

# --------------------------------------------------------------------- #
#  Helpers                                                                #
# --------------------------------------------------------------------- #
def download_file(url: str, save_path: str) -> None:
    """Stream-download a file from *url* to *save_path*."""
    print(f"  ↓ Downloading {url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"  ✓ Saved to {save_path}")


def parse_example(raw: dict) -> dict:
    """Normalize one QReCC JSON record into our schema."""
    context  = raw.get("Context", raw.get("context", []))
    question = raw.get("Question", raw.get("question", ""))
    answer   = raw.get("Answer", raw.get("answer", ""))
    if isinstance(context, str):
        context = [u.strip() for u in context.split("\n") if u.strip()]
    return {
        "context":   context,
        "question":  question,
        "answer":    answer,
        "num_turns": len(context),
    }


# --------------------------------------------------------------------- #
#  Main                                                                   #
# --------------------------------------------------------------------- #
def prepare_qrecc_data(smoke_test: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    train_json = os.path.join(DATA_DIR, "raw_train.json")
    test_json  = os.path.join(DATA_DIR, "raw_test.json")

    # 1. İndir (zaten varsa atla)
    if not os.path.exists(train_json):
        download_file(TRAIN_URL, train_json)
    if not os.path.exists(test_json):
        download_file(TEST_URL, test_json)

    # 2. Belleğe oku
    print("Loading JSON …")
    with open(train_json, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    if smoke_test:
        print("[SMOKE TEST] Veri 50 train / 20 test'e kısıtlandı.")
        train_raw = train_raw[:50]
        test_raw  = test_raw[:20]

    # 3. Parse
    parsed_train = [parse_example(r) for r in train_raw]
    parsed_test  = [parse_example(r) for r in test_raw]

    # 4. Filtre (3 ≤ turns ≤ 20)
    before = len(parsed_train)
    parsed_train = [ex for ex in parsed_train if 3 <= ex["num_turns"] <= 20]
    print(f"Train filtreleme: {before} → {len(parsed_train)}")

    # smoke-test güvenlik ağı
    if smoke_test and len(parsed_train) < 5:
        parsed_train = [parse_example(r) for r in train_raw[:10]]

    # 5. Kaydet
    Dataset.from_list(parsed_train).save_to_disk(os.path.join(DATA_DIR, "train_processed"))
    Dataset.from_list(parsed_test).save_to_disk(os.path.join(DATA_DIR, "test_processed"))
    print(f"✓ İşlenmiş veri kaydedildi → {DATA_DIR}")


if __name__ == "__main__":
    args = get_args()
    prepare_qrecc_data(smoke_test=args.smoke_test)
