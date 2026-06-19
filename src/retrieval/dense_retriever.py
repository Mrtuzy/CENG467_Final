"""Dense bi-encoder retriever (semantic) as a drop-in alternative to BM25.

Encodes passages and the query with a sentence-transformers bi-encoder and
ranks passages by cosine similarity. Shares the same interface as
``BM25Retriever`` (``index`` + ``retrieve``) so it can be swapped into the
evaluation runner without any other code change.
"""


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._passages: list[str] = []
        self._titles: list[str | None] = []
        self._passage_emb = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device is None:
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def index(self, passages: list[str], titles: list[str | None] = None) -> None:
        self._ensure_model()
        self._passages = passages
        self._titles = titles if titles is not None else [None] * len(passages)
        # Normalised embeddings so that a dot product equals cosine similarity.
        self._passage_emb = self._model.encode(
            passages,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self._passage_emb is None:
            raise RuntimeError("Call index() before retrieve().")
        q_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = self._passage_emb @ q_emb  # cosine similarity, shape (n_passages,)
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
    retriever = DenseRetriever()

    for sample in samples:
        passages = [" ".join(sents) for _, sents in sample["context"]]
        titles = [title for title, _ in sample["context"]]
        retriever.index(passages, titles)
        results = retriever.retrieve(sample["question"], k=5)
        print(f"Q: {sample['question'][:80]}")
        print(f"  Top-1: [{results[0]['title']}] score={results[0]['score']:.3f}")
        print()
