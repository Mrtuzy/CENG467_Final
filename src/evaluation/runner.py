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


class EvaluationRunner:
    def __init__(
        self,
        pruner_name: str,
        generator_model: str,
        judge_model: str,
        output_path: str,
    ):
        self.pruner_name = pruner_name
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, samples: list[dict], k_retrieve: int = 5) -> dict:
        """Run full pipeline and save results to JSONL."""
        pruner_class = PRUNER_REGISTRY.get(self.pruner_name)
        if pruner_class is None:
            raise ValueError(f"Unknown pruner: {self.pruner_name}")

        pruner = pruner_class()
        retriever = BM25Retriever()
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
        }

        for sample in tqdm(samples, desc=f"Running {self.pruner_name}"):
            # Retrieve
            passages = [" ".join(sents) for _, sents in sample["context"]]
            titles = [title for title, _ in sample["context"]]
            retriever.index(passages, titles)
            retrieved = retriever.retrieve(sample["question"], k=k_retrieve)

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
                "pruner": self.pruner_name,
                "faithfulness": judge_result["faithfulness"],
                "n_claims": judge_result["n_claims"],
                "em": em,
                "f1": f1,
                "compression_ratio": comp_ratio,
                "latency_s": gen_result["latency_s"],
            }
            results.append(result)

            metrics["faithfulness"].append(judge_result["faithfulness"])
            metrics["em"].append(em)
            metrics["f1"].append(f1)
            metrics["compression_ratio"].append(comp_ratio)
            metrics["latency_s"].append(gen_result["latency_s"])

        save_jsonl(results, str(self.output_path))

        aggregate = {
            "pruner": self.pruner_name,
            "n_samples": len(results),
            "faithfulness_mean": sum(metrics["faithfulness"]) / len(metrics["faithfulness"]),
            "em_mean": sum(metrics["em"]) / len(metrics["em"]),
            "f1_mean": sum(metrics["f1"]) / len(metrics["f1"]),
            "compression_ratio_mean": sum(metrics["compression_ratio"]) / len(metrics["compression_ratio"]),
            "latency_mean_s": sum(metrics["latency_s"]) / len(metrics["latency_s"]),
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
