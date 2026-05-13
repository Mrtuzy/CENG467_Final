from collections import Counter


def exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if pred equals gold (case-insensitive), else 0.0."""
    return 1.0 if pred.lower() == gold.lower() else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def bertscore(preds: list[str], golds: list[str]) -> dict:
    """Compute BERTScore and return P/R/F1."""
    if not preds or not golds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(preds, golds, lang="en", verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def compression_ratio(original_tokens: int, pruned_tokens: int) -> float:
    """Return ratio of pruned to original token count."""
    if original_tokens == 0:
        return 1.0
    return min(1.0, pruned_tokens / original_tokens)
