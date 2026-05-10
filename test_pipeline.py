#!/usr/bin/env python
"""Test SPEC-003, SPEC-004, SPEC-005 with dry_run mode and sample data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_jsonl
from src.pruning import PRUNER_REGISTRY
from src.generation.generator import RAGGenerator
from src.judge.judge import LLMJudge


def test_pruners():
    """Test all 3 baseline pruners."""
    print("=" * 60)
    print("TESTING SPEC-003: Baseline Pruners")
    print("=" * 60)

    sample_passages = [
        {
            "passage": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
            "title": "Eiffel Tower",
            "score": 0.8,
            "rank": 1,
        },
        {
            "passage": "Paris is the capital and most populous city of France with about 2.2 million residents.",
            "title": "Paris",
            "score": 0.7,
            "rank": 2,
        },
    ]
    query = "Where is the Eiffel Tower located?"

    for pruner_name, pruner_class in PRUNER_REGISTRY.items():
        pruner = pruner_class()
        result = pruner.prune(sample_passages, query, max_tokens=100)
        print(f"\n{pruner_name}:")
        print(f"  Output passages: {len(result)}")
        if result:
            print(f"  First passage (truncated): {result[0]['passage'][:100]}...")
        print(f"  Compression ratio: {pruner.compression_ratio:.2f}")


def test_generator():
    """Test dry_run generator."""
    print("\n" + "=" * 60)
    print("TESTING SPEC-004: Generator")
    print("=" * 60)

    gen = RAGGenerator("dry_run")

    context = [
        {"passage": "Paris is the capital of France."},
        {"passage": "The Eiffel Tower is in Paris."},
    ]
    question = "Where is the Eiffel Tower?"

    result = gen.generate(question, context)
    print(f"\nQuestion: {question}")
    print(f"Generated answer: {result['answer']}")
    print(f"Tokens - prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']}")
    print(f"Latency: {result['latency_s']:.4f}s")


def test_judge():
    """Test dry_run judge."""
    print("\n" + "=" * 60)
    print("TESTING SPEC-005: LLM-as-Judge")
    print("=" * 60)

    judge = LLMJudge("dry_run")

    answer = "Paris is in France and has the Eiffel Tower."
    context_passages = [
        {"passage": "Paris is the capital of France."},
        {"passage": "The Eiffel Tower is located in Paris, France."},
    ]

    result = judge.score(answer, context_passages)
    print(f"\nAnswer: {answer}")
    print(f"Faithfulness: {result['faithfulness']:.2f}")
    print(f"Claims: {result['n_claims']}")
    print(f"Supported: {result['supported']}/{result['n_claims']}")
    for i, claim in enumerate(result['claims'], 1):
        status = 'PASS' if claim['supported'] else 'FAIL'
        print(f"  {i}. {claim['claim']} [{status}]")


def test_pipeline():
    """Test mini pipeline with real data."""
    print("\n" + "=" * 60)
    print("TESTING MINI PIPELINE (dry_run mode)")
    print("=" * 60)

    from src.retrieval.bm25_retriever import BM25Retriever

    samples = load_jsonl("data/processed/hotpotqa_experiments.jsonl")[:2]

    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Q: {sample['question'][:80]}")

        # Retrieve
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever = BM25Retriever()
        retriever.index(passages, titles)
        retrieved = retriever.retrieve(sample["question"], k=3)
        print(f"Retrieved: {len(retrieved)} passages")

        # Prune
        pruner = PRUNER_REGISTRY["recomp"]()
        pruned = pruner.prune(retrieved, sample["question"], max_tokens=200)
        print(f"After pruning: {len(pruned)} passages, ratio={pruner.compression_ratio:.2f}")

        # Generate
        gen = RAGGenerator("dry_run")
        gen_result = gen.generate(sample["question"], pruned)
        print(f"Generated: {gen_result['answer']}")

        # Judge
        judge = LLMJudge("dry_run")
        judge_result = judge.score(gen_result["answer"], pruned)
        print(f"Faithfulness: {judge_result['faithfulness']:.2f}")


if __name__ == "__main__":
    test_pruners()
    test_generator()
    test_judge()
    test_pipeline()
    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)
