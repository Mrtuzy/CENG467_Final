from abc import ABC, abstractmethod


class BasePruner(ABC):
    def __init__(self):
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

    @abstractmethod
    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        """Return pruned passages. Each dict must have: passage, title, score, rank, pruned."""

    @property
    def compression_ratio(self) -> float:
        if self._last_input_tokens == 0:
            return 1.0
        return self._last_output_tokens / self._last_input_tokens

    def _count_tokens(self, text: str) -> int:
        return len(text.split())
