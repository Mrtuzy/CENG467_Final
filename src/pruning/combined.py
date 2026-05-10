from .base import BasePruner
from .provence import ProvencePruner
from .history_pruning import HistoryPruner


class CombinedPruner(BasePruner):
    """Chains Provence then History pruning."""

    def __init__(self, history: list[str] = None):
        super().__init__()
        self.history = history or []
        self.provence = ProvencePruner()
        self.history_pruner = HistoryPruner(history)

    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        # First pass: Provence
        provence_out = self.provence.prune(passages, query, max_tokens)
        input_tokens = self.provence._last_input_tokens

        # Second pass: History (on Provence output)
        combined_out = self.history_pruner.prune(provence_out, query, max_tokens)
        output_tokens = self.history_pruner._last_output_tokens

        self._last_input_tokens = input_tokens
        self._last_output_tokens = output_tokens

        return combined_out
