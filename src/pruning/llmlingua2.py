from .base import BasePruner


class LLMLingua2Pruner(BasePruner):
    """Token-level compression using LLMLingua-2."""

    def __init__(
        self,
        compression_rate: float = 0.5,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    ):
        super().__init__()
        self.compression_rate = compression_rate
        self.model_name = model_name
        self._compressor = None
        self.fallback_used = False

    def _load_compressor(self) -> None:
        if self._compressor is not None:
            return
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
            )
        except ImportError:
            pass

    def prune(self, passages: list[dict], query: str, max_tokens: int = 512) -> list[dict]:
        combined = " ".join(p["passage"] for p in passages)
        self._last_input_tokens = len(combined.split())

        if self._compressor is None:
            self._load_compressor()

        if self._compressor is None:
            # Fallback to RECOMP if llmlingua not available
            self.fallback_used = True
            from .recomp import RecompPruner
            fallback = RecompPruner()
            result = fallback.prune(passages, query, max_tokens)
            self._last_output_tokens = sum(len(p["passage"].split()) for p in result)
            return result

        self.fallback_used = False
        compressed_result = self._compressor.compress_prompt(
            combined,
            rate=self.compression_rate,
            force_tokens=["\n", ".", "?", "!"],
        )
        if isinstance(compressed_result, dict):
            compressed = (
                compressed_result.get("compressed_prompt")
                or compressed_result.get("compressed")
                or ""
            )
        else:
            compressed = str(compressed_result)
        if not compressed:
            raise RuntimeError("LLMLingua-2 returned an empty compressed prompt.")
        self._last_output_tokens = len(compressed.split())

        return [{
            "passage": compressed,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
