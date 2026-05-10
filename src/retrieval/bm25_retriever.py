from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, tokenizer: str = "whitespace"):
        self.tokenizer = tokenizer
        self._bm25 = None
        self._passages: list[str] = []
        self._titles: list[str | None] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, passages: list[str], titles: list[str | None] = None) -> None:
        self._passages = passages
        self._titles = titles if titles is not None else [None] * len(passages)
        tokenized = [self._tokenize(p) for p in passages]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve().")
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        k = min(k, len(self._passages))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "passage": self._passages[idx],
                "title": self._titles[idx],
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.io import load_jsonl

    samples = load_jsonl("data/processed/hotpotqa_experiments.jsonl")[:5]
    retriever = BM25Retriever()

    for sample in samples:
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        results = retriever.retrieve(sample["question"], k=5)
        print(f"Q: {sample['question'][:80]}")
        print(f"  Top-1: [{results[0]['title']}] score={results[0]['score']:.3f}")
        print()
