"""Run all 7 pruning strategies with a single shared model instance."""
import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.io import load_jsonl, save_jsonl
from src.retrieval.bm25_retriever import BM25Retriever
from src.pruning import PRUNER_REGISTRY
from src.generation.generator import RAGGenerator
from src.judge.judge import LLMJudge
from src.evaluation.metrics import exact_match, token_f1, compression_ratio

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
N_SAMPLES = 50
DATA_PATH = "data/processed/hotpotqa_experiments.jsonl"
RESULTS_DIR = "experiments/results"

PRUNERS = [
    "no_pruning",
    "naive_truncation",
    "recomp",
    "llmlingua2",
    "provence",
    "history_pruning",
    "combined",
    "lnn_sentence",
    "lnn_token",
]

PRUNER_LABELS = {
    "no_pruning":       "B1: No Pruning",
    "naive_truncation": "B2: Naive Truncation",
    "recomp":           "B3: RECOMP",
    "llmlingua2":       "S1: LLMLingua-2",
    "provence":         "S2: Provence",
    "history_pruning":  "S3: History Pruning",
    "combined":         "S4: Combined",
    "lnn_sentence":     "S5a-v2: Q-LiSP",
    "lnn_token":        "S5b: LNN Token",
}


def run_pruner(pruner_name, samples, generator, judge, output_path):
    pruner = PRUNER_REGISTRY[pruner_name]()
    retriever = BM25Retriever()
    results = []
    metrics = {"faithfulness": [], "em": [], "f1": [], "compression_ratio": [], "latency_s": []}

    for sample in tqdm(samples, desc=f"  {PRUNER_LABELS[pruner_name]}"):
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        retrieved = retriever.retrieve(sample["question"], k=5)

        input_tokens = sum(len(p["passage"].split()) for p in retrieved)
        pruned = pruner.prune(retrieved, sample["question"], max_tokens=512)
        output_tokens = sum(len(p["passage"].split()) for p in pruned)

        gen_result = generator.generate(sample["question"], pruned)
        judge_result = judge.score(gen_result["answer"], pruned)

        em = exact_match(gen_result["answer"], sample["answer"])
        f1 = token_f1(gen_result["answer"], sample["answer"])
        comp = compression_ratio(input_tokens, output_tokens)

        result = {
            "sample_id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "generated_answer": gen_result["answer"],
            "pruner": pruner_name,
            "faithfulness": judge_result["faithfulness"],
            "n_claims": judge_result["n_claims"],
            "em": em,
            "f1": f1,
            "compression_ratio": comp,
            "latency_s": gen_result["latency_s"],
        }
        results.append(result)
        for k in metrics:
            metrics[k].append(result[k])

    save_jsonl(results, output_path)

    n = len(results)
    return {
        "pruner": pruner_name,
        "n_samples": n,
        "faithfulness_mean": sum(metrics["faithfulness"]) / n,
        "em_mean": sum(metrics["em"]) / n,
        "f1_mean": sum(metrics["f1"]) / n,
        "compression_ratio_mean": sum(metrics["compression_ratio"]) / n,
        "latency_mean_s": sum(metrics["latency_s"]) / n,
    }


def main():
    print(f"Loading {N_SAMPLES} samples ...")
    samples = load_jsonl(DATA_PATH)[:N_SAMPLES]
    print(f"Loaded {len(samples)} samples.")

    # Load model ONCE and share between generator and judge
    print(f"\nLoading {MODEL} (once for all strategies) ...")
    generator = RAGGenerator(MODEL)
    generator.load_model()
    judge = LLMJudge(MODEL)
    judge._model = generator._model
    judge._tokenizer = generator._tokenizer
    print("Model ready.\n")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    aggregates = []

    for pruner_name in PRUNERS:
        output_path = f"{RESULTS_DIR}/{pruner_name}_{N_SAMPLES}samples.jsonl"
        print(f"\n{'='*65}")
        print(f"  Pruner : {PRUNER_LABELS[pruner_name]}")
        print(f"  Output : {output_path}")
        print(f"{'='*65}")

        agg = run_pruner(pruner_name, samples, generator, judge, output_path)
        aggregates.append(agg)

        print(f"  Faithfulness : {agg['faithfulness_mean']:.3f}")
        print(f"  EM           : {agg['em_mean']:.3f}")
        print(f"  F1           : {agg['f1_mean']:.3f}")
        print(f"  Comp. Ratio  : {agg['compression_ratio_mean']:.3f}")
        print(f"  Latency(s)   : {agg['latency_mean_s']:.2f}")

    summary_path = f"{RESULTS_DIR}/aggregate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)
    print(f"\nAggregate summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<26} {'Faithful':>9} {'EM':>7} {'F1':>7} {'CompRatio':>10} {'Lat(s)':>8}")
    print("-" * 80)
    for agg in aggregates:
        label = PRUNER_LABELS.get(agg["pruner"], agg["pruner"])
        print(
            f"{label:<26} "
            f"{agg['faithfulness_mean']:>9.3f} "
            f"{agg['em_mean']:>7.3f} "
            f"{agg['f1_mean']:>7.3f} "
            f"{agg['compression_ratio_mean']:>10.3f} "
            f"{agg['latency_mean_s']:>8.2f}"
        )
    print("=" * 80)
    print(f"\nAll results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
