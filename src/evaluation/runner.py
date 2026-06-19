import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.retrieval.bm25_retriever import BM25Retriever
from src.pruning import PRUNER_REGISTRY
from src.generation.generator import RAGGenerator
from src.judge.judge import LLMJudge
from src.utils.io import load_jsonl, save_jsonl
from src.evaluation.metrics import exact_match, token_f1, bertscore, compression_ratio


def _retrieval_metrics(gold_titles: set, retrieved_titles: list) -> tuple[float, float]:
    """Recall and reciprocal rank of gold supporting passages among retrieved.

    Recall = fraction of gold supporting titles present in the retrieved list.
    MRR    = reciprocal rank of the first retrieved gold supporting passage.
    """
    if not gold_titles:
        return 0.0, 0.0
    retrieved_set = set(retrieved_titles)
    recall = len(gold_titles & retrieved_set) / len(gold_titles)
    rr = 0.0
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in gold_titles:
            rr = 1.0 / rank
            break
    return recall, rr


class EvaluationRunner:
    def __init__(
        self,
        pruner_name: str,
        generator_model: str,
        judge_model: str,
        output_path: str,
        retriever=None,
        retriever_name: str = "bm25",
    ):
        self.pruner_name = pruner_name
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Inject any retriever exposing index()/retrieve(); defaults to BM25 so
        # existing callers keep working unchanged.
        self.retriever = retriever
        self.retriever_name = retriever_name

    def run(self, samples: list[dict], k_retrieve: int = 5) -> dict:
        """Run full pipeline and save results to JSONL."""
        pruner_class = PRUNER_REGISTRY.get(self.pruner_name)
        if pruner_class is None:
            raise ValueError(f"Unknown pruner: {self.pruner_name}")

        pruner = pruner_class()
        retriever = self.retriever if self.retriever is not None else BM25Retriever()
        generator = RAGGenerator(self.generator_model)
        judge = LLMJudge(self.judge_model)

        if self.generator_model != "dry_run":
            generator.load_model()
            # Share loaded model with judge to avoid loading 14GB twice
            if self.judge_model == self.generator_model:
                judge._model = generator._model
                judge._tokenizer = generator._tokenizer

        results = []
        metrics = {
            "faithfulness": [],
            "em": [],
            "f1": [],
            "compression_ratio": [],
            "latency_s": [],
            "retrieval_recall": [],
            "retrieval_mrr": [],
        }

        for sample in tqdm(samples, desc=f"Running {self.retriever_name}/{self.pruner_name}"):
            # Retrieve
            passages = [" ".join(sents) for _, sents in sample["context"]]
            titles = [title for title, _ in sample["context"]]
            retriever.index(passages, titles)
            retrieved = retriever.retrieve(sample["question"], k=k_retrieve)

            # Retrieval-quality metrics against HotpotQA gold supporting titles.
            gold_titles = {t for t, _ in sample.get("supporting_facts", [])}
            retrieved_titles = [p["title"] for p in retrieved]
            recall, mrr = _retrieval_metrics(gold_titles, retrieved_titles)

            # Calculate input tokens
            input_tokens = sum(len(p["passage"].split()) for p in retrieved)

            # Prune
            pruned = pruner.prune(retrieved, sample["question"], max_tokens=512)
            output_tokens = sum(len(p["passage"].split()) for p in pruned)

            # Generate
            gen_result = generator.generate(sample["question"], pruned)

            # Evaluate
            judge_result = judge.score(gen_result["answer"], pruned)
            em = exact_match(gen_result["answer"], sample["answer"])
            f1 = token_f1(gen_result["answer"], sample["answer"])
            comp_ratio = compression_ratio(input_tokens, output_tokens)

            result = {
                "sample_id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "generated_answer": gen_result["answer"],
                "retriever": self.retriever_name,
                "pruner": self.pruner_name,
                "faithfulness": judge_result["faithfulness"],
                "n_claims": judge_result["n_claims"],
                "em": em,
                "f1": f1,
                "compression_ratio": comp_ratio,
                "latency_s": gen_result["latency_s"],
                "retrieval_recall": recall,
                "retrieval_mrr": mrr,
            }
            results.append(result)

            metrics["faithfulness"].append(judge_result["faithfulness"])
            metrics["em"].append(em)
            metrics["f1"].append(f1)
            metrics["compression_ratio"].append(comp_ratio)
            metrics["latency_s"].append(gen_result["latency_s"])
            metrics["retrieval_recall"].append(recall)
            metrics["retrieval_mrr"].append(mrr)

        save_jsonl(results, str(self.output_path))

        aggregate = {
            "retriever": self.retriever_name,
            "pruner": self.pruner_name,
            "n_samples": len(results),
            "faithfulness_mean": sum(metrics["faithfulness"]) / len(metrics["faithfulness"]),
            "em_mean": sum(metrics["em"]) / len(metrics["em"]),
            "f1_mean": sum(metrics["f1"]) / len(metrics["f1"]),
            "compression_ratio_mean": sum(metrics["compression_ratio"]) / len(metrics["compression_ratio"]),
            "latency_mean_s": sum(metrics["latency_s"]) / len(metrics["latency_s"]),
            "retrieval_recall_mean": sum(metrics["retrieval_recall"]) / len(metrics["retrieval_recall"]),
            "retrieval_mrr_mean": sum(metrics["retrieval_mrr"]) / len(metrics["retrieval_mrr"]),
        }

        return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument("--pruner", type=str, default="no_pruning", help="Pruner strategy")
    parser.add_argument("--generator", type=str, default="dry_run", help="Generator model")
    parser.add_argument("--judge", type=str, default="dry_run", help="Judge model")
    parser.add_argument("--data", type=str, default="data/processed/hotpotqa_experiments.jsonl", help="Data path")
    parser.add_argument("--n_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output", type=str, default="experiments/results/eval.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    if args.n_samples:
        samples = samples[: args.n_samples]

    runner = EvaluationRunner(args.pruner, args.generator, args.judge, args.output)
    aggregate = runner.run(samples)

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")
    print("=" * 60)
    print(f"Pruner: {aggregate['pruner']}")
    print(f"Samples: {aggregate['n_samples']}")
    print(f"Faithfulness: {aggregate['faithfulness_mean']:.3f}")
    print(f"EM: {aggregate['em_mean']:.3f}")
    print(f"F1: {aggregate['f1_mean']:.3f}")
    print(f"Compression: {aggregate['compression_ratio_mean']:.3f}")
    print(f"Latency: {aggregate['latency_mean_s']:.3f}s")
