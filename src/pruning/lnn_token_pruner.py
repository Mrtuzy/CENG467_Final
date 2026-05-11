"""S5b — Token-level LNN pruner using Liquid Time-Constant (LTC) RNN.

Processes token sequences through an LTC-RNN to produce per-token importance
scores, enabling fine-grained context compression with temporal awareness.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

from .base import BasePruner
from .lnn_utils import split_sentences, SentenceEncoder


class LNNTokenModel(nn.Module):
    """Token importance scorer built on an LTC sequence model.

    Architecture:
        input (token_emb ⊕ query_emb ⊕ turn_feat)
          → Linear projection (input_dim → proj_dim)
          → LTC RNN (proj_dim → output_dim, return_sequences=True)
          → Linear (output_dim → 1)
          → Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 769,   # 384 + 384 + 1
        proj_dim: int = 128,
        hidden_units: int = 128,
        output_units: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        from ncps.wirings import AutoNCP
        from ncps.torch import LTC

        wiring = AutoNCP(units=hidden_units, output_size=output_units)
        self.ltc_rnn = LTC(in_features=proj_dim, units=wiring, return_sequences=True)

        self.scorer = nn.Sequential(
            nn.Linear(output_units, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    @property
    def state_size(self):
        return self.ltc_rnn.state_size if hasattr(self.ltc_rnn, 'state_size') else 0

    def forward(
        self,
        features: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (1, seq_len, input_dim)
            hx: optional hidden state from previous turn

        Returns:
            scores: (seq_len,)
            hx_new: updated hidden state
        """
        projected = self.projection(features)       # (1, seq_len, proj_dim)
        output = self.ltc_rnn(projected, hx=hx)

        # ncps may return (output,) or (output, hx)
        if isinstance(output, tuple):
            rnn_out, hx_new = output[0], output[1] if len(output) > 1 else None
        else:
            rnn_out, hx_new = output, None

        scores = torch.sigmoid(self.scorer(rnn_out))  # (1, seq_len, 1)
        scores = scores.squeeze(0).squeeze(-1)         # (seq_len,)
        return scores, hx_new


class LNNTokenPruner(BasePruner):
    """S5b pruner: uses a trained LNNTokenModel for per-token importance."""

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
        self._model: Optional[LNNTokenModel] = None
        self._hidden_state = None
        self._turn_index: int = 0
        self._model_path = model_path

    def _load_model(self):
        if self._model is not None:
            return

        if self._model_path:
            path = Path(self._model_path)
        else:
            path = Path(__file__).parent.parent.parent / "models" / "lnn_token.pt"

        self.encoder.load()
        emb_dim = self.encoder.embedding_dim
        input_dim = emb_dim * 2 + 1

        self._model = LNNTokenModel(input_dim=input_dim)

        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)

        self._model.to(self.device)
        self._model.eval()

    def reset_state(self):
        """Reset hidden state between conversations."""
        self._hidden_state = None
        self._turn_index = 0

    def prune(
        self, passages: list[dict], query: str, max_tokens: int = 512
    ) -> list[dict]:
        self._load_model()

        # Concatenate all passage text
        full_text = " ".join(p["passage"] for p in passages if p["passage"].strip())
        if not full_text.strip():
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            return [{"passage": "", "title": None, "score": 0.0, "rank": 1, "pruned": True}]

        words = full_text.split()
        self._last_input_tokens = len(words)

        # Get token embeddings
        tokens, token_embs = self.encoder.encode_tokens(full_text, max_length=256)
        seq_len = token_embs.shape[0]

        query_emb = self.encoder.encode([query])[0]
        q_broadcast = np.tile(query_emb, (seq_len, 1))
        turn_feat = np.full((seq_len, 1), self._turn_index / 10.0)
        features = np.concatenate([token_embs, q_broadcast, turn_feat], axis=1)

        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores, self._hidden_state = self._model(features_t, self._hidden_state)
            scores = scores.cpu().numpy()  # (seq_len,)

        self._turn_index += 1

        # Reconstruct text keeping high-importance tokens
        tokenizer = self.encoder._model.tokenizer
        enc = tokenizer(
            full_text, truncation=True, max_length=256,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]

        # Collect token spans above threshold
        kept_chars = set()
        # Sort by score descending, pick within budget
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        token_budget = max_tokens
        count = 0
        for tok_idx, score in indexed_scores:
            if score < self.threshold and count > 0:
                continue
            if tok_idx >= len(offsets):
                continue
            start, end = offsets[tok_idx]
            if start == end:
                continue
            # Approximate word count for this token
            tok_text = full_text[start:end]
            n_words = max(1, len(tok_text.split()))
            if count + n_words > token_budget:
                break
            for c in range(start, end):
                kept_chars.add(c)
            count += n_words

        # Reconstruct: keep words where majority of chars are kept
        result_words = []
        pos = 0
        for word in words:
            word_start = full_text.find(word, pos)
            if word_start == -1:
                pos += len(word) + 1
                continue
            word_end = word_start + len(word)
            kept_count = sum(1 for c in range(word_start, word_end) if c in kept_chars)
            if kept_count > len(word) * 0.5:
                result_words.append(word)
            pos = word_end + 1

        self._last_output_tokens = len(result_words)
        retained_text = " ".join(result_words)

        return [{
            "passage": retained_text,
            "title": None,
            "score": 0.0,
            "rank": 1,
            "pruned": True,
        }]
