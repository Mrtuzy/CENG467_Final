# Judge Calibration

This directory contains human-annotated samples for calibrating the LLM-as-Judge faithfulness scorer against human annotations.

## Annotation Format

Each sample in `hotpotqa_calibration_raw.jsonl` is annotated with a `human_faithfulness` score. Create a separate annotation file `human_labels.jsonl` with the following format:

```json
{
  "sample_id": "5a8b57f25542995d1e6f1371",
  "answer": "yes",
  "context": "Scott Derrickson is an American director. Ed Wood was an American filmmaker.",
  "human_faithfulness": 1.0,
  "notes": "Both claims are clearly supported by the context."
}
```

### Fields

- **sample_id** (string): Unique identifier matching a sample in `hotpotqa_calibration_raw.jsonl`
- **answer** (string): The generated answer being evaluated
- **context** (string): The retrieved context passages concatenated
- **human_faithfulness** (float): Score from 0.0 (no claims supported) to 1.0 (all claims supported)
- **notes** (string, optional): Annotator comments explaining the score

## Calibration Process

Run calibration after collecting human labels:

```python
from src.utils.io import load_jsonl
from src.judge.judge import LLMJudge
from src.judge.calibration import run_calibration

samples = load_jsonl("data/processed/hotpotqa_calibration.jsonl")
human_labels = load_jsonl("data/judge_calibration/human_labels.jsonl")
judge = LLMJudge("dry_run")
result = run_calibration(judge, samples, human_labels)
print(f"Cohen's κ: {result['cohen_kappa']:.3f}")
```

Results are saved to `calibration_results.json`.
