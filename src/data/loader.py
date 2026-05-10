import json
import argparse
from pathlib import Path


def _convert_context(raw_context: dict) -> list:
    """Convert HuggingFace HotpotQA context dict to list of [title, sentences] pairs."""
    titles = raw_context["title"]
    sentences = raw_context["sentences"]
    return [[title, sents] for title, sents in zip(titles, sentences)]


def _convert_supporting_facts(raw_sf: dict) -> list:
    """Convert HuggingFace supporting_facts dict to list of [title, sent_id] pairs."""
    titles = raw_sf["title"]
    sent_ids = raw_sf["sent_id"]
    return [[title, sid] for title, sid in zip(titles, sent_ids)]


def load_hotpotqa(split: str, n_samples: int = None) -> list[dict]:
    """Load HotpotQA (distractor setting) from HuggingFace datasets.

    Saves raw output to data/raw/hotpotqa_{split}.jsonl and returns samples.
    Each sample has: id, question, answer, context, supporting_facts.
    """
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    samples = []
    for item in dataset:
        sample = {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "context": _convert_context(item["context"]),
            "supporting_facts": _convert_supporting_facts(item["supporting_facts"]),
        }
        samples.append(sample)
        if n_samples is not None and len(samples) >= n_samples:
            break

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"hotpotqa_{split}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess HotpotQA dataset")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to load")
    parser.add_argument("--n_samples", type=int, default=None, help="Max number of samples")
    args = parser.parse_args()

    samples = load_hotpotqa(args.split, args.n_samples)
    print(f"Loaded {len(samples)} samples from HotpotQA {args.split} split")
    print(f"Saved to data/raw/hotpotqa_{args.split}.jsonl")
