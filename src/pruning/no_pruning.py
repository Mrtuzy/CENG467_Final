from .base import BasePruner


class NoPruningPruner(BasePruner):
    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        total = sum(self._count_tokens(p["passage"]) for p in passages)
        self._last_input_tokens = total
        self._last_output_tokens = total
        return [{**p, "pruned": False} for p in passages]

    @property
    def compression_ratio(self) -> float:
        return 1.0
