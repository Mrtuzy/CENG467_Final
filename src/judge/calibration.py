import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from .judge import LLMJudge


def run_calibration(
    judge: LLMJudge,
    calibration_samples: list[dict],
    human_labels: list[dict] = None,
) -> dict:
    """Run judge on calibration samples and compute Cohen's κ vs human labels.

    If human_labels not provided, uses a dummy set for dry runs.
    """
    judge_scores = []
    human_scores = []

    for i, sample in enumerate(calibration_samples):
        context_passages = [
            {"passage": " ".join(sents)} for _, sents in sample.get("context", [])
        ]
        result = judge.score(sample["answer"], context_passages)
        judge_scores.append(result["faithfulness"])

        if human_labels and i < len(human_labels):
            human_scores.append(human_labels[i].get("human_faithfulness", 0.5))
        else:
            human_scores.append(0.5)

    judge_binary = [1 if s > 0.5 else 0 for s in judge_scores]
    human_binary = [1 if s > 0.5 else 0 for s in human_scores]
    kappa = cohen_kappa_score(human_binary, judge_binary) if len(set(human_binary + judge_binary)) > 1 else 1.0

    result = {
        "cohen_kappa": float(kappa),
        "judge_scores": judge_scores,
        "human_scores": human_scores,
    }

    out_dir = Path("data/judge_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
