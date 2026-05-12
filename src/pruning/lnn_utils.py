"""Shared utilities for LNN-based pruning (S5a / S5b)."""

import re
import torch
import numpy as np
from typing import Optional

LNN_SENTENCE_SCALAR_FEATURES = 7


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def lnn_sentence_input_dim(embedding_dim: int) -> int:
    """Return the Q-LiSP sentence feature width for a given embedding size."""
    return embedding_dim * 4 + LNN_SENTENCE_SCALAR_FEATURES


def build_query_aware_sentence_features(
    sent_embs: np.ndarray,
    query_emb: np.ndarray,
    *,
    token_counts: Optional[list[int] | np.ndarray] = None,
    ranks: Optional[list[float] | np.ndarray] = None,
    retriever_scores: Optional[list[float] | np.ndarray] = None,
    turn_index: int = 0,
) -> np.ndarray:
    """Build Q-LiSP V2 sentence features.

    Features are:
        sent_emb, query_emb, sent_emb * query_emb, abs(sent_emb - query_emb),
        cosine, position, recency, length_norm, rank_norm, retriever_score,
        turn_norm.
    """
    sent_embs = np.asarray(sent_embs, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    n = sent_embs.shape[0]
    if n == 0:
        return np.empty((0, lnn_sentence_input_dim(query_emb.shape[0])), dtype=np.float32)

    q_broadcast = np.tile(query_emb, (n, 1)).astype(np.float32)
    prod = sent_embs * q_broadcast
    abs_diff = np.abs(sent_embs - q_broadcast)

    denom = (
        np.linalg.norm(sent_embs, axis=1, keepdims=True)
        * np.linalg.norm(q_broadcast, axis=1, keepdims=True)
        + 1e-8
    )
    cosine = np.sum(sent_embs * q_broadcast, axis=1, keepdims=True) / denom

    positions = (np.arange(n, dtype=np.float32).reshape(-1, 1) / max(n - 1, 1))
    recency = 1.0 - positions

    if token_counts is None:
        token_counts_arr = np.ones((n, 1), dtype=np.float32)
    else:
        token_counts_arr = np.asarray(token_counts, dtype=np.float32).reshape(-1, 1)
    length_norm = token_counts_arr / max(float(token_counts_arr.max()), 1.0)

    if ranks is None:
        ranks_arr = np.ones((n, 1), dtype=np.float32)
    else:
        ranks_arr = np.asarray(ranks, dtype=np.float32).reshape(-1, 1)
    rank_norm = 1.0 / np.maximum(ranks_arr, 1.0)

    if retriever_scores is None:
        retriever_scores_arr = np.zeros((n, 1), dtype=np.float32)
    else:
        retriever_scores_arr = np.asarray(retriever_scores, dtype=np.float32).reshape(-1, 1)

    turn_norm = np.full((n, 1), turn_index / 10.0, dtype=np.float32)

    return np.concatenate(
        [
            sent_embs,
            q_broadcast,
            prod,
            abs_diff,
            cosine.astype(np.float32),
            positions,
            recency,
            length_norm,
            rank_norm,
            retriever_scores_arr,
            turn_norm,
        ],
        axis=1,
    ).astype(np.float32)


class SentenceEncoder:
    """Wraps sentence-transformers for sentence & token embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def embedding_dim(self) -> int:
        self.load()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self.load()
        return self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def encode_tokens(self, text: str, max_length: int = 256):
        """Return (tokens, token_embeddings) using the underlying transformer."""
        self.load()
        tokenizer = self._model.tokenizer
        auto_model = self._model[0].auto_model

        encoded = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        )
        with torch.no_grad():
            outputs = auto_model(**{k: v.to(self.device) for k, v in encoded.items()})
        token_embs = outputs.last_hidden_state[0].cpu().numpy()  # (seq_len, hidden)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return tokens, token_embs


class MultiTurnSimulator:
    """Creates multi-turn conversations from HotpotQA single-turn samples."""

    def __init__(self, turns_per_conversation: int = 3, decay_factor: float = 0.5):
        self.turns_per_conversation = turns_per_conversation
        self.decay_factor = decay_factor

    def create_conversations(self, samples: list[dict]) -> list[list[dict]]:
        convs = []
        for i in range(0, len(samples), self.turns_per_conversation):
            group = samples[i : i + self.turns_per_conversation]
            if len(group) >= 2:
                convs.append(group)
        return convs

    def get_supporting_set(self, sample: dict) -> set[tuple[str, int]]:
        sf = sample.get("supporting_facts", {})
        titles = sf.get("title", [])
        idxs = sf.get("sent_id", sf.get("sent_idx", []))
        return {(t, i) for t, i in zip(titles, idxs)}

    def sentence_labels_for_turn(
        self, conversation: list[dict], turn_idx: int
    ) -> dict:
        """Return supporting facts + decayed historical facts for a turn."""
        current_sf = self.get_supporting_set(conversation[turn_idx])
        historical: dict[tuple[str, int], float] = {}
        for prev in range(turn_idx):
            prev_sf = self.get_supporting_set(conversation[prev])
            dist = turn_idx - prev
            decay = self.decay_factor ** dist
            for key in prev_sf:
                if key not in historical or historical[key] < decay:
                    historical[key] = decay
        return {
            "supporting_facts": current_sf,
            "historical_facts": historical,
            "turn_index": turn_idx,
        }

    def label_sentence(
        self, title: str, sent_id: int, turn_info: dict
    ) -> float:
        key = (title, sent_id)
        if key in turn_info["supporting_facts"]:
            return 1.0
        if key in turn_info["historical_facts"]:
            return turn_info["historical_facts"][key]
        return 0.0


def build_sentence_dataset(
    samples: list[dict],
    encoder: SentenceEncoder,
    simulator: MultiTurnSimulator,
    max_sentences_per_sample: int = 60,
):
    """Build training tensors for sentence-level LNN.

    Returns dict with keys:
        features  : list of (n_sents, feat_dim) tensors   (per-conversation-turn)
        labels    : list of (n_sents,) tensors
        turn_idxs : list of int
    """
    conversations = simulator.create_conversations(samples)
    all_features, all_labels, all_turns = [], [], []

    for conv in conversations:
        # Encode all queries once
        queries = [s["question"] for s in conv]
        query_embs = encoder.encode(queries)

        prev_hidden = None  # not used at data-build time, kept for API compat

        for turn_idx, sample in enumerate(conv):
            turn_info = simulator.sentence_labels_for_turn(conv, turn_idx)
            q_emb = query_embs[turn_idx]

            sents, labels, titles_idxs = [], [], []
            for title, sent_list in sample["context"]:
                for si, sent_text in enumerate(sent_list):
                    sents.append(sent_text)
                    labels.append(simulator.label_sentence(title, si, turn_info))
                    titles_idxs.append((title, si))

            if not sents:
                continue

            sents = sents[:max_sentences_per_sample]
            labels = labels[:max_sentences_per_sample]
            titles_idxs = titles_idxs[:max_sentences_per_sample]

            positive_keys = set(turn_info["supporting_facts"])
            for i, (title, sent_id) in enumerate(titles_idxs):
                if labels[i] > 0.0:
                    continue
                if (title, sent_id - 1) in positive_keys or (title, sent_id + 1) in positive_keys:
                    labels[i] = 0.4

            sent_embs = encoder.encode(sents)  # (n, emb_dim)
            token_counts = [len(s.split()) for s in sents]
            features = build_query_aware_sentence_features(
                sent_embs,
                q_emb,
                token_counts=token_counts,
                ranks=np.ones(len(sents), dtype=np.float32),
                retriever_scores=np.zeros(len(sents), dtype=np.float32),
                turn_index=turn_idx,
            )

            all_features.append(torch.tensor(features, dtype=torch.float32))
            all_labels.append(torch.tensor(labels, dtype=torch.float32))
            all_turns.append(turn_idx)

    return {"features": all_features, "labels": all_labels, "turn_idxs": all_turns}


def build_token_dataset(
    samples: list[dict],
    encoder: SentenceEncoder,
    simulator: MultiTurnSimulator,
    max_tokens_per_sample: int = 256,
):
    """Build training tensors for token-level LNN.

    Returns dict with keys:
        features  : list of (seq_len, feat_dim) tensors
        labels    : list of (seq_len,) tensors
        turn_idxs : list of int
    """
    conversations = simulator.create_conversations(samples)
    all_features, all_labels, all_turns = [], [], []

    for conv in conversations:
        query_embs = encoder.encode([s["question"] for s in conv])

        for turn_idx, sample in enumerate(conv):
            turn_info = simulator.sentence_labels_for_turn(conv, turn_idx)
            q_emb = query_embs[turn_idx]

            # Build full passage text and per-char label map
            full_text_parts, char_labels = [], []
            for title, sent_list in sample["context"]:
                for si, sent_text in enumerate(sent_list):
                    lbl = simulator.label_sentence(title, si, turn_info)
                    full_text_parts.append(sent_text)
                    char_labels.extend([lbl] * len(sent_text))
                    char_labels.append(lbl)  # space separator

            full_text = " ".join(full_text_parts)
            if not full_text.strip():
                continue

            tokens, token_embs = encoder.encode_tokens(
                full_text, max_length=max_tokens_per_sample
            )

            # Map token labels from char labels (approximate via position)
            tokenizer = encoder._model.tokenizer
            enc = tokenizer(
                full_text, truncation=True, max_length=max_tokens_per_sample,
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]
            tok_labels = []
            for start, end in offsets:
                if start == end:
                    tok_labels.append(0.0)
                else:
                    mid = (start + end) // 2
                    if mid < len(char_labels):
                        tok_labels.append(char_labels[mid])
                    else:
                        tok_labels.append(0.0)

            seq_len = token_embs.shape[0]
            q_broadcast = np.tile(q_emb, (seq_len, 1))
            turn_feat = np.full((seq_len, 1), turn_idx / 10.0)
            features = np.concatenate([token_embs, q_broadcast, turn_feat], axis=1)

            all_features.append(torch.tensor(features, dtype=torch.float32))
            all_labels.append(torch.tensor(tok_labels[:seq_len], dtype=torch.float32))
            all_turns.append(turn_idx)

    return {"features": all_features, "labels": all_labels, "turn_idxs": all_turns}
