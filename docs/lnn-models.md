# LNN Models: Architectures and Pipelines

This document summarizes the LNN-based pruners in this repo (S5a and S5b),
their architectures, and the training/evaluation pipelines.

## High-level overview

- **S5a (Sentence-level LNN)** scores sentences for pruning.
- **S5b (Token-level LNN)** scores tokens for pruning.
- Both use **Liquid Time-Constant (LTC) cells** to model time-varying
  importance across multi-turn conversations.

## Model architectures

### S5a: Sentence-level LNN (LNNSentenceModel)

**Inputs (per turn):**

- Sentence embeddings for all sentences in retrieved passages.
- Query embedding for the current question, broadcast to each sentence.
- A scalar turn feature (turn index / 10.0).

**Feature vector per sentence:**

```
[sentence_emb | query_emb | turn_feat]  ->  input_dim = emb_dim * 2 + 1
```

**Architecture:**

1. Linear projection + LayerNorm + GELU
2. LTC cell (AutoNCP wiring)
3. MLP scorer + sigmoid to produce sentence importance in [0, 1]

**Temporal modeling:**

- Sentences are processed **sequentially** by the LTC cell.
- A hidden state is carried **across turns** within a simulated conversation.
- The hidden state is reset at turn 0.

### S5b: Token-level LNN (LNNTokenModel)

**Inputs (per turn):**

- Token embeddings for a concatenated passage (truncated to max length).
- Query embedding, broadcast to each token.
- A scalar turn feature (turn index / 10.0).

**Feature vector per token:**

```
[token_emb | query_emb | turn_feat]  ->  input_dim = emb_dim * 2 + 1
```

**Architecture:**

- Same idea as S5a but applied to token sequences.
- LTC cell processes the token sequence and outputs per-token scores.

## Training pipelines

### S5a training (sentence-level)

1. **Load HotpotQA** validation split.
2. **Simulate multi-turn conversations** using a fixed number of turns
	per conversation (default 3).
3. **Label sentences**:
	- Current supporting facts -> label 1.0
	- Historical supporting facts -> decayed label (e.g., 0.5, 0.25)
4. **Build features**: sentence embeddings + query embeddings + turn feature.
5. **Train with BCELoss** over sentence scores.
6. **Save best model** to `models/lnn_sentence.pt`.

Artifacts:

- `models/lnn_sentence.pt`
- `models/training_history.json`
- `models/training_curves.png`

### S5b training (token-level)

1. Same conversation simulation and label logic.
2. Build **token-level labels** by mapping sentence labels to token offsets.
3. Build features: token embeddings + query embeddings + turn feature.
4. Train with BCELoss on token scores.
5. Save model to `models/lnn_token.pt`.

Artifacts:

- `models/lnn_token.pt`
- `models/training_history.json`
- `models/training_curves.png`

## Inference pipelines

### S5a pruning pipeline

1. **Retrieve passages** (e.g., BM25 in evaluation scripts).
2. **Split into sentences**.
3. **Encode** sentences and query with the shared sentence encoder.
4. **Score sentences** with the trained LNN model.
5. **Select sentences** by score until a token budget is reached.
6. **Return pruned context** (concatenated selected sentences).

### S5b pruning pipeline

1. **Retrieve passages**.
2. **Concatenate passage text** and tokenize.
3. **Encode tokens** and query.
4. **Score tokens** with the trained LNN model.
5. **Select tokens** by score (or threshold) and reconstruct pruned text.

## Where these are wired in code

- Sentence model: `src/pruning/lnn_sentence_pruner.py`
- Token model: `src/pruning/lnn_token_pruner.py`
- Training loops: `src/pruning/lnn_trainer.py`
- Shared utilities: `src/pruning/lnn_utils.py`

## Notes

- The same sentence encoder is reused across S5a and S5b to reduce memory.
- Hidden state is carried within a conversation to model time dynamics.
- If running in Colab, outputs can be copied to Drive via the provided
  notebook cells.
