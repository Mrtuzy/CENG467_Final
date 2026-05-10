import re
from .base import BasePruner


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class ProvencePruner(BasePruner):
    """Sentence-level pruning using semantic similarity."""

    def __init__(self, relevance_threshold: float = 0.6):
        super().__init__()
        self.relevance_threshold = relevance_threshold
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except (ImportError, Exception):
            self._model = None

    def _cosine_sim(self, a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        from .recomp import _split_sentences as recomp_split

        self._load_model()

        # Collect all sentences with scores
        scored_sentences: list[tuple[float, str]] = []
        for p in passages:
            for sent in _split_sentences(p["passage"]):
                if self._model is not None:
                    try:
                        query_emb = self._model.encode(query)
                        sent_emb = self._model.encode(sent)
                        score = float(self._cosine_sim(query_emb, sent_emb))
                    except Exception:
                        score = 0.5
                else:
                    # Fallback: Jaccard similarity
                    query_tokens = set(query.lower().split())
                    sent_tokens = set(sent.lower().split())
                    if not query_tokens and not sent_tokens:
                        score = 0.0
                    elif not query_tokens or not sent_tokens:
                        score = 0.0
                    else:
                        score = len(query_tokens & sent_tokens) / len(query_tokens | sent_tokens)

                scored_sentences.append((score, sent))

        self._last_input_tokens = sum(self._count_tokens(s) for _, s in scored_sentences)

        # Keep sentences above threshold, or top 60% if none above threshold
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        threshold = max(self.relevance_threshold, sorted([s for s, _ in scored_sentences], reverse=True)[len(scored_sentences) // 2] if scored_sentences else 0)

        kept: list[str] = []
        token_count = 0
        for score, sent in scored_sentences:
            if score < self.relevance_threshold:
                continue
            n = self._count_tokens(sent)
            if token_count + n > max_tokens:
                break
            kept.append(sent)
            token_count += n

        self._last_output_tokens = token_count
        retained_text = " ".join(kept)

        return [{
            "passage": retained_text,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
