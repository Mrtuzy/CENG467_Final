import json
import random
from pathlib import Path

from src.data.loader import load_hotpotqa


def make_splits(n_total: int, val_ratio: float = 0.1) -> dict:
    """Split indices into train and val sets.

    Returns {"train": [indices], "val": [indices]} for the given total.
    """
    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    n_val = max(1, int(n_total * val_ratio))
    return {
        "val": sorted(all_indices[:n_val]),
        "train": sorted(all_indices[n_val:]),
    }


def make_calibration_set(n: int = 50) -> list[dict]:
    """Load n random samples from HotpotQA train split for judge calibration.

    Saves output to data/judge_calibration/hotpotqa_calibration_raw.jsonl.
    Returns list of sample dicts.
    """
    samples = load_hotpotqa("train", n_samples=n)

    cal_dir = Path("data/judge_calibration")
    cal_dir.mkdir(parents=True, exist_ok=True)
    out_path = cal_dir / "hotpotqa_calibration_raw.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples


def prepare_processed_data(
    n_experiments: int = 200,
    n_calibration: int = 50,
) -> None:
    """Download and save experiment + calibration JSONL files to data/processed/."""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n_experiments} experiment samples from validation split...")
    exp_samples = load_hotpotqa("validation", n_samples=n_experiments)
    exp_path = processed_dir / "hotpotqa_experiments.jsonl"
    with open(exp_path, "w", encoding="utf-8") as f:
        for sample in exp_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(exp_samples)} samples to {exp_path}")

    print(f"Loading {n_calibration} calibration samples from train split...")
    cal_samples = load_hotpotqa("train", n_samples=n_calibration)
    cal_path = processed_dir / "hotpotqa_calibration.jsonl"
    with open(cal_path, "w", encoding="utf-8") as f:
        for sample in cal_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(cal_samples)} samples to {cal_path}")

    ids_exp = [s["id"] for s in exp_samples]
    assert len(ids_exp) == len(set(ids_exp)), "Duplicate IDs found in experiments set"
    ids_cal = [s["id"] for s in cal_samples]
    assert len(ids_cal) == len(set(ids_cal)), "Duplicate IDs found in calibration set"

    print("All checks passed. Data preparation complete.")


if __name__ == "__main__":
    prepare_processed_data()
