"""Coverage and noise metrics for pruning evaluation.

These metrics use HotpotQA's supporting_facts annotations to measure:
  - Coverage:  how many gold supporting sentences survive pruning
  - Noise:     what fraction of retained content is non-supporting
  - Trade-off: combined score balancing coverage vs noise
"""

import re
from typing import Optional


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def supporting_fact_coverage(
    pruned_passages: list[dict],
    sample: dict,
) -> float:
    """Fraction of gold supporting sentences retained after pruning.

    Args:
        pruned_passages: output of a pruner's prune() method
        sample: HotpotQA sample dict with 'context' and 'supporting_facts'

    Returns:
        coverage in [0, 1]
    """
    sf = sample.get("supporting_facts", {})
    titles = sf.get("title", [])
    idxs = sf.get("sent_id", [])

    if not titles:
        return 1.0

    # Collect gold supporting sentences text
    context_map: dict[str, list[str]] = {}
    for title, sents in sample["context"]:
        context_map[title] = sents

    gold_sents = []
    for title, idx in zip(titles, idxs):
        if title in context_map and idx < len(context_map[title]):
            gold_sents.append(_normalize(context_map[title][idx]))

    if not gold_sents:
        return 1.0

    # Check how many gold sentences appear in pruned output
    pruned_text = _normalize(" ".join(p["passage"] for p in pruned_passages))
    found = 0
    for gs in gold_sents:
        # Substring match (pruned text may have minor formatting differences)
        if gs in pruned_text:
            found += 1
        else:
            # Fuzzy: check if >80% of words are present
            gs_words = set(gs.split())
            pruned_words = set(pruned_text.split())
            if gs_words and len(gs_words & pruned_words) / len(gs_words) > 0.8:
                found += 1

    return found / len(gold_sents)


def noise_ratio(
    pruned_passages: list[dict],
    sample: dict,
) -> float:
    """Fraction of retained tokens that come from non-supporting sentences.

    Lower is better. Returns value in [0, 1].

    Args:
        pruned_passages: output of pruner
        sample: HotpotQA sample with 'context' and 'supporting_facts'

    Returns:
        noise ratio in [0, 1]
    """
    sf = sample.get("supporting_facts", {})
    titles = sf.get("title", [])
    idxs = sf.get("sent_id", [])

    # Build set of supporting sentence words
    context_map: dict[str, list[str]] = {}
    for title, sents in sample["context"]:
        context_map[title] = sents

    supporting_words: set[str] = set()
    for title, idx in zip(titles, idxs):
        if title in context_map and idx < len(context_map[title]):
            supporting_words.update(context_map[title][idx].lower().split())

    pruned_text = " ".join(p["passage"] for p in pruned_passages)
    pruned_words = pruned_text.lower().split()

    if not pruned_words:
        return 0.0

    noise_count = sum(1 for w in pruned_words if w not in supporting_words)
    return noise_count / len(pruned_words)


def coverage_noise_f1(coverage: float, noise: float) -> float:
    """F1-like score combining coverage (recall) and purity (1 - noise).

    coverage_noise_f1 = 2 * coverage * purity / (coverage + purity)
    where purity = 1 - noise
    """
    purity = 1.0 - noise
    if coverage + purity == 0:
        return 0.0
    return 2.0 * coverage * purity / (coverage + purity)


def coverage_noise_report(
    pruned_passages: list[dict],
    sample: dict,
) -> dict:
    """Full coverage-noise report for a single sample."""
    cov = supporting_fact_coverage(pruned_passages, sample)
    noise = noise_ratio(pruned_passages, sample)
    f1 = coverage_noise_f1(cov, noise)
    return {
        "coverage": cov,
        "noise_ratio": noise,
        "purity": 1.0 - noise,
        "coverage_noise_f1": f1,
    }
