"""Cross-encoder reranker retriever as a drop-in alternative to BM25.

Instead of encoding query and passages independently (bi-encoder) or matching
lexical tokens (BM25), a cross-encoder jointly attends over the (query,
passage) pair and outputs a single relevance score. Because each HotpotQA
question carries only ten candidate passages, the cross-encoder can score all
candidates directly, acting as a strong single-stage retriever.

Shares the same interface as ``BM25Retriever`` (``index`` + ``retrieve``).
"""


class CrossEncoderRetriever:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._passages: list[str] = []
        self._titles: list[str | None] = []

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = self.device
            if device is None:
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def index(self, passages: list[str], titles: list[str | None] = None) -> None:
        # Nothing to precompute: a cross-encoder needs the query, so all scoring
        # happens at retrieval time. We just store the candidate pool.
        self._ensure_model()
        self._passages = passages
        self._titles = titles if titles is not None else [None] * len(passages)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if not self._passages:
            raise RuntimeError("Call index() before retrieve().")
        pairs = [[query, passage] for passage in self._passages]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        k = min(k, len(self._passages))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "passage": self._passages[idx],
                    "title": self._titles[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.io import load_jsonl

    samples = load_jsonl("data/processed/hotpotqa_experiments.jsonl")[:5]
    retriever = CrossEncoderRetriever()

    for sample in samples:
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        results = retriever.retrieve(sample["question"], k=5)
        print(f"Q: {sample['question'][:80]}")
        print(f"  Top-1: [{results[0]['title']}] score={results[0]['score']:.3f}")
        print()
