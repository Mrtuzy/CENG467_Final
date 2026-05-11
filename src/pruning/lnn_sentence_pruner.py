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
from .lnn_utils import split_sentences, SentenceEncoder


class LNNSentenceModel(nn.Module):
    """Sentence importance scorer built on an LTC cell.

    Architecture:
        input (sent_emb ⊕ query_emb ⊕ turn_feat)
          → Linear projection (input_dim → proj_dim)
          → LTCCell (proj_dim → hidden_dim, output_dim)
          → Linear (output_dim → 1)
          → Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 769,    # 384 + 384 + 1
        proj_dim: int = 128,
        hidden_units: int = 64,
        output_units: int = 32,
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
        )

        # LTC cell via ncps
        from ncps.wirings import AutoNCP
        from ncps.torch import LTCCell

        wiring = AutoNCP(units=hidden_units, output_size=output_units)
        self.ltc_cell = LTCCell(wiring, in_features=proj_dim)
        self.state_size = self.ltc_cell.state_size

        self.scorer = nn.Sequential(
            nn.Linear(output_units, 16),
            nn.GELU(),
            nn.Linear(16, 1),
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
            scores: (n_sentences,) — importance scores in [0, 1]
            hx_new: (1, state_size) — updated hidden state
        """
        n = features.size(0)
        projected = self.projection(features)  # (n, proj_dim)

        if hx is None:
            hx = torch.zeros(1, self.state_size, device=features.device)

        scores = []
        for i in range(n):
            x_i = projected[i : i + 1]  # (1, proj_dim)
            output, hx = self.ltc_cell(x_i, hx)  # output: (1, output_units)
            score = torch.sigmoid(self.scorer(output))  # (1, 1)
            scores.append(score.squeeze(-1))

        scores = torch.cat(scores, dim=0)  # (n,)
        return scores, hx


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
        input_dim = emb_dim * 2 + 1  # sent_emb + query_emb + turn_feat

        self._model = LNNSentenceModel(input_dim=input_dim)

        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)

        self._model.to(self.device)
        self._model.eval()

    def reset_state(self):
        """Reset hidden state (call between conversations)."""
        self._hidden_state = None
        self._turn_index = 0

    def prune(
        self, passages: list[dict], query: str, max_tokens: int = 512
    ) -> list[dict]:
        self._load_model()

        # Collect all sentences with metadata
        all_sents = []
        for p in passages:
            for sent in split_sentences(p["passage"]):
                all_sents.append(sent)

        if not all_sents:
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            return [{"passage": "", "title": None, "score": 0.0, "rank": 1, "pruned": True}]

        self._last_input_tokens = sum(self._count_tokens(s) for s in all_sents)

        # Encode
        sent_embs = self.encoder.encode(all_sents)
        query_emb = self.encoder.encode([query])[0]
        q_broadcast = np.tile(query_emb, (len(all_sents), 1))
        turn_feat = np.full((len(all_sents), 1), self._turn_index / 10.0)
        features = np.concatenate([sent_embs, q_broadcast, turn_feat], axis=1)
        features_t = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Score
        with torch.no_grad():
            scores, self._hidden_state = self._model(features_t, self._hidden_state)
            scores = scores.cpu().numpy()

        self._turn_index += 1

        # Select sentences above threshold, respecting token budget
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        kept = []
        token_count = 0
        for idx, score in indexed:
            if score < self.threshold and kept:
                continue
            n = self._count_tokens(all_sents[idx])
            if token_count + n > max_tokens:
                break
            kept.append((idx, all_sents[idx], float(score)))
            token_count += n

        # Restore original order
        kept.sort(key=lambda x: x[0])
        self._last_output_tokens = token_count

        retained_text = " ".join(s for _, s, _ in kept)
        return [{
            "passage": retained_text,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
