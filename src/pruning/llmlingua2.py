from .base import BasePruner


class LLMLingua2Pruner(BasePruner):
    """Token-level compression using LLMLingua-2."""

    def __init__(self, compression_rate: float = 0.5):
        super().__init__()
        self.compression_rate = compression_rate
        self._compressor = None

    def _load_compressor(self) -> None:
        if self._compressor is not None:
            return
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor()
        except ImportError:
            pass

    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        combined = " ".join(p["passage"] for p in passages)
        self._last_input_tokens = len(combined.split())

        if self._compressor is None:
            self._load_compressor()

        if self._compressor is None:
            # Fallback to RECOMP if llmlingua not available
            from .recomp import RecompPruner
            fallback = RecompPruner()
            result = fallback.prune(passages, query, max_tokens)
            self._last_output_tokens = sum(len(p["passage"].split()) for p in result)
            return result

        compressed = self._compressor.compress_prompt(
            combined,
            target_token=int(self._last_input_tokens * self.compression_rate),
        )
        self._last_output_tokens = len(compressed.split())

        return [{
            "passage": compressed,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
