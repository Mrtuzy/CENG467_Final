# RAG Faithfulness Pruning

**Evaluating Context and History Pruning Strategies for RAG Faithfulness in Small Language Models via LLM-as-Judge**

Course: CENG 467 — Natural Language Understanding and Generation

Team:
- Mert Güden — 300201013
- Berkay Fehmi Tekin — 300201090

---

## Dataset

This project uses **HotpotQA** (distractor setting), a multi-hop QA dataset.

### Automatic Download

The dataset is downloaded automatically from HuggingFace when you run the pipeline.

```bash
# Download and prepare 200 experiment samples + 50 calibration samples
python src/data/splitter.py

# Or download a custom split manually
python src/data/loader.py --split validation --n_samples 200
```

Files will be saved to:
- `data/raw/hotpotqa_validation.jsonl` — raw downloaded data
- `data/processed/hotpotqa_experiments.jsonl` — 200 samples for evaluation
- `data/processed/hotpotqa_calibration.jsonl` — 50 samples for judge calibration

### Manual Download

If you prefer to download manually, visit [HotpotQA](https://hotpotqa.github.io) and place the JSON files in `data/raw/`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Run full evaluation for one strategy
python src/evaluation/runner.py --pruner no_pruning --output experiments/results/b1_no_pruning.jsonl

# Compare all baselines
python src/evaluation/runner.py --all --output experiments/results/
```

---

## Pruning Strategies

| ID | Strategy | Description |
|----|----------|-------------|
| B1 | No Pruning | Full top-5 context |
| B2 | Naive Truncation | Hard 512-token cutoff |
| B3 | RECOMP Extractive | Query-relevant sentence selection |
| S1 | LLMLingua-2 | Token-level compression |
| S2 | Provence | Sentence-level pruning |
| S3 | Semantic History Pruning | Conversation-history aware pruning |
| S4 | Provence + History | Combined sentence + history pruning |
