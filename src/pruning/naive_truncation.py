from .base import BasePruner


class NaiveTruncationPruner(BasePruner):
    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        combined = " ".join(p["passage"] for p in passages)
        tokens = combined.split()
        self._last_input_tokens = len(tokens)
        truncated_tokens = tokens[:max_tokens]
        self._last_output_tokens = len(truncated_tokens)
        truncated_text = " ".join(truncated_tokens)
        return [{
            "passage": truncated_text,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
