import re
from .base import BasePruner


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class RecompPruner(BasePruner):
    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        query_tokens = set(query.lower().split())

        scored_sentences: list[tuple[float, str]] = []
        for p in passages:
            for sent in _split_sentences(p["passage"]):
                sent_tokens = set(sent.lower().split())
                score = _jaccard(query_tokens, sent_tokens)
                scored_sentences.append((score, sent))

        self._last_input_tokens = sum(self._count_tokens(s) for _, s in scored_sentences)

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        kept: list[str] = []
        token_count = 0
        for _, sent in scored_sentences:
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
