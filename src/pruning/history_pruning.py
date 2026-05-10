import re
from .base import BasePruner


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a or b) else 0.0


class HistoryPruner(BasePruner):
    """Sentence-level pruning aware of conversation history."""

    def __init__(self, history: list[str] = None):
        super().__init__()
        self.history = history or []

    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        if not self.history:
            # No history: pass through like no_pruning
            total = sum(self._count_tokens(p["passage"]) for p in passages)
            self._last_input_tokens = total
            self._last_output_tokens = total
            return [{**p, "pruned": False} for p in passages]

        # Score sentences by redundancy with history
        history_text = " ".join(self.history)
        history_tokens = set(history_text.lower().split())

        scored_sentences: list[tuple[float, str]] = []
        for p in passages:
            for sent in _split_sentences(p["passage"]):
                sent_tokens = set(sent.lower().split())
                # Inverse score: higher score = less redundant (more novel)
                redundancy = _jaccard(history_tokens, sent_tokens)
                novelty = 1.0 - redundancy
                scored_sentences.append((novelty, sent))

        self._last_input_tokens = sum(self._count_tokens(s) for _, s in scored_sentences)

        # Keep sentences sorted by novelty (descending)
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
