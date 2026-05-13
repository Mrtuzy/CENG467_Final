# RAG Faithfulness Pruning

**Evaluating Context and History Pruning Strategies for RAG Faithfulness in Small Language Models via LLM-as-Judge**

Course: CENG 467 — Natural Language Understanding and Generation

Team:
- Mert Güden — 300201013
- Berkay Fehmi Tekin — 300201090

Repository: https://github.com/Mrtuzy/CENG467_Final

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

## Colab Reproducibility

The experiments are designed to be run in Google Colab rather than on a local laptop.
Use the notebooks below as the primary reproduction path:

1. `notebooks/01_data_exploration.ipynb` — dataset loading and sanity checks
2. `notebooks/02_baseline_results.ipynb` — baseline dry-run / small-scale checks
3. `notebooks/05_final_colab_runs.ipynb` — final gap-closing runs:
   - real LLMLingua-2 with the `llmlingua` package installed
   - synthetic two-turn history pruning for S3/S4
   - LLM-as-Judge calibration template and Cohen's kappa computation
   - qualitative error-analysis candidate export
4. `notebooks/03_lnn_training_colab.ipynb` and `notebooks/04_lnn_evaluation_colab.ipynb` — optional LNN/CfC-oriented exploratory extension

Open `05_final_colab_runs.ipynb` in a GPU Colab runtime and run the cells from top to bottom.
It clones this repository, installs dependencies, downloads HotpotQA, and writes final outputs to `experiments/results/`.

## Local Smoke Tests

```bash
# Optional local sanity check only; final reported runs are produced in Colab notebooks.
python test_pipeline.py
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
