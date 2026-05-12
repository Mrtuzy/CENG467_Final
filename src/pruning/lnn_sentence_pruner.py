"""S5a — Sentence-level LNN pruner using Liquid Time-Constant (LTC) cells.

The LTC cell processes sentences sequentially, maintaining a hidden state that
captures temporal dynamics of token importance across conversation turns.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

from .base import BasePruner
from .lnn_utils import (
    build_query_aware_sentence_features,
    lnn_sentence_input_dim,
    split_sentences,
    SentenceEncoder,
)


class LNNSentenceModel(nn.Module):
    """Query-aware sentence importance scorer built on an LTC cell.

    Architecture:
        input (query-aware Q-LiSP features)
          → Linear projection (input_dim → proj_dim)
          → LTCCell (proj_dim → hidden_dim, output_dim)
          → Residual scorer (output_dim ⊕ proj_dim → 1)
          → Logits
    """

    def __init__(
        self,
        input_dim: int = 1543,    # 384 * 4 + 7 for all-MiniLM-L6-v2
        proj_dim: int = 128,
        hidden_units: int = 64,
        output_units: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LTC cell via ncps
        from ncps.wirings import AutoNCP
        from ncps.torch import LTCCell

        wiring = AutoNCP(units=hidden_units, output_size=output_units)
        self.ltc_cell = LTCCell(wiring, in_features=proj_dim)
        self.state_size = self.ltc_cell.state_size

        self.scorer = nn.Sequential(
            nn.Linear(output_units + proj_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (n_sentences, input_dim) — all sentences for one turn
            hx: (1, state_size) — hidden state from previous turn, or None

        Returns:
            logits: (n_sentences,) — raw sentence importance logits
            hx_new: (1, state_size) — updated hidden state
        """
        n = features.size(0)
        projected = self.projection(features)  # (n, proj_dim)

        if hx is None:
            hx = torch.zeros(1, self.state_size, device=features.device)

        logits = []
        for i in range(n):
            x_i = projected[i : i + 1]  # (1, proj_dim)
            output, hx = self.ltc_cell(x_i, hx)  # output: (1, output_units)
            score_input = torch.cat([output, x_i], dim=-1)
            logit = self.scorer(score_input)  # (1, 1)
            logits.append(logit.squeeze(-1))

        logits = torch.cat(logits, dim=0)  # (n,)
        return logits, hx


class LNNSentencePruner(BasePruner):
    """S5a pruner: uses a trained LNNSentenceModel to score sentences."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.4,
        device: str = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.encoder = SentenceEncoder(device=self.device)
        self._model: Optional[LNNSentenceModel] = None
        self._hidden_state: Optional[torch.Tensor] = None
        self._turn_index: int = 0
        self._model_path = model_path

    def _load_model(self):
        if self._model is not None:
            return

        # Determine model path
        if self._model_path:
            path = Path(self._model_path)
        else:
            path = Path(__file__).parent.parent.parent / "models" / "lnn_sentence.pt"

        self.encoder.load()
        emb_dim = self.encoder.embedding_dim
        input_dim = lnn_sentence_input_dim(emb_dim)

        self._model = LNNSentenceModel(input_dim=input_dim)

        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            try:
                self._model.load_state_dict(state)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Could not load lnn_sentence weights. Q-LiSP V2 uses a new "
                    "feature/model format and requires retraining."
                ) from exc

        self._model.to(self.device)
        self._model.eval()

    def reset_state(self):
        """Reset hidden state (call between conversations)."""
        self._hidden_state = None
        self._turn_index = 0

    def _collect_sentence_items(self, passages: list[dict]) -> list[dict]:
        items = []
        for p_idx, passage in enumerate(passages):
            sentences = split_sentences(passage.get("passage", ""))
            rank = passage.get("rank", p_idx + 1)
            score = passage.get("score", 0.0)
            for s_idx, sent in enumerate(sentences):
                items.append({
                    "text": sent,
                    "passage_idx": p_idx,
                    "sent_idx": s_idx,
                    "title": passage.get("title"),
                    "rank": float(rank or p_idx + 1),
                    "retriever_score": float(score or 0.0),
                    "token_count": self._count_tokens(sent),
                })
        return items

    @staticmethod
    def _smooth_scores(scores: np.ndarray, items: list[dict]) -> np.ndarray:
        if len(scores) <= 1:
            return scores
        smoothed = np.zeros_like(scores, dtype=np.float32)
        for i, score in enumerate(scores):
            left = scores[i - 1] if i > 0 and items[i - 1]["passage_idx"] == items[i]["passage_idx"] else score
            right = (
                scores[i + 1]
                if i + 1 < len(scores) and items[i + 1]["passage_idx"] == items[i]["passage_idx"]
                else score
            )
            smoothed[i] = 0.2 * left + 0.6 * score + 0.2 * right
        return smoothed

    def _build_spans(self, scores: np.ndarray, items: list[dict]) -> list[dict]:
        spans = []
        start = None
        for i, score in enumerate(scores):
            boundary = i > 0 and items[i]["passage_idx"] != items[i - 1]["passage_idx"]
            if boundary and start is not None:
                spans.append(self._make_span(start, i - 1, scores, items))
                start = None
            if score >= self.threshold and start is None:
                start = i
            elif score < self.threshold and start is not None:
                spans.append(self._make_span(start, i - 1, scores, items))
                start = None

        if start is not None:
            spans.append(self._make_span(start, len(scores) - 1, scores, items))
        return spans

    @staticmethod
    def _make_span(start: int, end: int, scores: np.ndarray, items: list[dict]) -> dict:
        span_scores = scores[start : end + 1]
        token_count = sum(item["token_count"] for item in items[start : end + 1])
        span_score = float(span_scores.mean() + 0.1 * span_scores.max())
        return {
            "start": start,
            "end": end,
            "score": span_score,
            "value": float(span_score / np.sqrt(max(token_count, 1))),
            "token_count": token_count,
        }

    def _select_spans(self, spans: list[dict], items: list[dict], scores: np.ndarray, max_tokens: int) -> list[dict]:
        selected = []
        token_count = 0
        for span in sorted(spans, key=lambda s: s["value"], reverse=True):
            if span["token_count"] > max_tokens:
                continue
            if token_count + span["token_count"] <= max_tokens:
                selected.append(span)
                token_count += span["token_count"]

        if selected:
            return sorted(selected, key=lambda s: s["start"])

        fallback = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        for idx in fallback:
            if items[idx]["token_count"] <= max_tokens:
                return [self._make_span(idx, idx, scores, items)]
        return []

    def prune(
        self, passages: list[dict], query: str, max_tokens: int = 512
    ) -> list[dict]:
        self._load_model()

        all_items = self._collect_sentence_items(passages)

        if not all_items:
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            return [{"passage": "", "title": None, "score": 0.0, "rank": 1, "pruned": True}]

        self._last_input_tokens = sum(item["token_count"] for item in all_items)

        # Encode
        texts = [item["text"] for item in all_items]
        sent_embs = self.encoder.encode(texts)
        query_emb = self.encoder.encode([query])[0]
        features = build_query_aware_sentence_features(
            sent_embs,
            query_emb,
            token_counts=[item["token_count"] for item in all_items],
            ranks=[item["rank"] for item in all_items],
            retriever_scores=[item["retriever_score"] for item in all_items],
            turn_index=self._turn_index,
        )
        features_t = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Score
        with torch.no_grad():
            logits, self._hidden_state = self._model(features_t, self._hidden_state)
            scores = torch.sigmoid(logits).cpu().numpy()

        self._turn_index += 1

        scores = self._smooth_scores(scores.astype(np.float32), all_items)
        spans = self._build_spans(scores, all_items)
        selected_spans = self._select_spans(spans, all_items, scores, max_tokens)

        kept = []
        for span in selected_spans:
            for idx in range(span["start"], span["end"] + 1):
                kept.append((idx, all_items[idx], float(scores[idx])))

        self._last_output_tokens = sum(item["token_count"] for _, item, _ in kept)

        retained_text = " ".join(item["text"] for _, item, _ in kept)
        avg_score = float(np.mean([score for _, _, score in kept])) if kept else 0.0
        debug = [
            {
                "index": idx,
                "text": item["text"],
                "score": score,
                "title": item["title"],
                "passage_idx": item["passage_idx"],
                "sent_idx": item["sent_idx"],
            }
            for idx, item, score in kept
        ]
        return [{
            "passage": retained_text,
            "title": None,
            "score": avg_score,
            "rank": 1,
            "pruned": True,
            "metadata": {
                "kept_sentences": debug,
                "selected_spans": selected_spans,
                "input_tokens": self._last_input_tokens,
                "output_tokens": self._last_output_tokens,
                "compression_ratio": self.compression_ratio,
            },
        }]
