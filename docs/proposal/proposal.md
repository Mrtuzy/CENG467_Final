CENG 467 — Natural Language Understanding and Generation
İzmir Institute of Technology  |  Spring 2026
Custom Project Proposal
 

Faithfulness Under Pressure:
Evaluating Context and History Pruning Strategies
for RAG Faithfulness in Small Language Models via LLM-as-Judge

1. Problem Description

Retrieval-Augmented Generation (RAG) has become the dominant architecture for grounding language model responses in external knowledge. However, faithfulness — the degree to which a generated answer is factually supported by the retrieved context — remains a critical open problem, especially for small language models in the 1–7B parameter range.

When retrieved passages are long or multi-document, small models are prone to (a) hallucinating facts not present in the context, (b) over-relying on parametric memory instead of the provided evidence, and (c) losing focus on relevant spans due to the "lost-in-the-middle" phenomenon (Liu et al., 2024). Context pruning addresses these failure modes by filtering out irrelevant sentences or tokens before generation.

Simultaneously, in multi-turn or agentic settings, conversation history accumulates rapidly. Naïve history retention inflates the effective context length and introduces stale or contradictory cues, further degrading faithfulness.

This project investigates how different pruning strategies — applied at the token level, the sentence level, and the conversation-history level — affect conversational RAG faithfulness for 1–7B models (Mistral 7B, Llama 3.1 8B). The main benchmark is QReCC, since it directly exposes the history-dependent question answering setting targeted by our pruning method. Faithfulness is evaluated using a systematic LLM-as-Judge pipeline calibrated against human annotations, closing the gap between automated metrics and human judgment.

2. Proposed Approach

The project constructs a unified RAG evaluation harness with three interchangeable pruning modules plugged between the retriever and the generator:

2.1  Pruning Strategies Compared

Strategy
Method
Reference
Token-level compression
LLMLingua-2: token classification with bidirectional encoder (XLM-RoBERTa-large), 2–5× compression ratio
Pan et al., ACL 2024
Sentence-level extractive pruning
Provence: DeBERTa-v3 dual-head model performing simultaneous reranking + sentence-level pruning; zero additional latency
Chirkova et al., ICLR 2025
History pruning
Semantic rolling-window: embed history turns, compute cosine similarity to current query, drop turns below threshold τ; keep last-k turns as minimum
Ablation over τ and k
Combination
Sentence-level pruning (Provence) + history pruning applied jointly
Proposed combination
 

2.2  LLM-as-Judge Faithfulness Pipeline

Following RAGAS (Es et al., 2023) and DeepEval's FaithfulnessMetric, the judge decomposes each generated answer into atomic claims, then verifies each claim against the (pruned or unpruned) retrieved context using a judge model (Mistral 7B Instruct). This produces a claim-level faithfulness score in [0, 1]. For calibration, a sample of 200 examples will be annotated by two human raters; inter-annotator agreement (Cohen's κ) will be reported.

2.3  Pipeline Architecture

Conversational turn + history → Optional question rewrite → Dense Retriever (FAISS + sentence-transformers) → [Context/History Pruner Module] → Generator (Mistral 7B / Llama 3.1 8B) → LLM-as-Judge → Faithfulness Score. All components are swappable; the retriever and generator are held fixed across conditions so that only the pruning intervention varies.

3. Baseline Methods

Baseline 1 — No Pruning (Full Context RAG)

The standard RAG pipeline without any context reduction. Top-5 retrieved passages are concatenated and passed directly to the generator. This serves as the upper-bound for context richness but lower-bound for faithfulness in noise-heavy retrieval sets. Metrics provide the reference point for measuring faithfulness degradation or improvement under pruning.

Baseline 2 — Truncation-Based Context Reduction

Passages are truncated to a fixed token budget (e.g., 512 tokens) by hard cut-off, independent of semantic relevance to the query. This naive baseline simulates real-world token-budget enforcement without intelligent selection. It is commonly used in production RAG systems and represents the most obvious (but uninformed) alternative to no pruning.

Baseline 3 — RECOMP Extractive Compressor

RECOMP (Xu et al., ACL 2024) independently encodes sentences and selects those with embeddings closest to the query. Unlike Provence, it assumes a single relevant sentence per passage and does not perform reranking. Comparing RECOMP to Provence isolates the benefit of Provence's adaptive, multi-sentence selection and its unified reranking-pruning head. This baseline directly reflects prior work on extractive compression and enables a fair architectural comparison.

4. Dataset / Benchmark

Dataset
Description
Usage
QReCC
Conversational open-domain QA with 14K conversations, about 81K QA pairs, question rewrites, answer provenance, and a 54M-passage web collection. Best fit for history pruning. (Anantha et al., 2021)
Primary evaluation
HotpotQA (distractor)
Multi-hop QA with 10 retrieved paragraphs (2 gold + 8 distractors). Used only as a pilot/sanity benchmark for the current pruning pipeline. (Yang et al., 2018)
Pilot implementation checks
TruthfulQA
857 questions designed to elicit model hallucinations. Used to probe whether pruning reduces parametric over-reliance. (Lin et al., 2022)
Hallucination stress-test
 

For QReCC, documents will be retrieved from the official QReCC passage collection (10M web pages split into 54M passages). HotpotQA outputs in the repository should be treated as pilot runs, not the final benchmark.

5. Evaluation Strategy & Metrics

Metric
Measures
Tool / Reference
Faithfulness (LLM-Judge)
Fraction of atomic claims supported by retrieved context
Custom pipeline; RAGAS framework
Exact Match (EM) & F1
Answer correctness against gold answers
Standard QReCC / QA eval scripts
BERTScore F1
Semantic similarity of generated answer to ground truth
bert-score library
Context compression ratio
Token reduction achieved by pruner (higher = more compression)
Token count ratio
Latency (ms/query)
End-to-end inference time per query
Python time.perf_counter
Judge–Human Agreement (κ)
Calibration of LLM judge against human annotations
Cohen's κ on 200-sample subset
 

Ablation studies will vary: (a) compression ratio τ for LLMLingua-2, (b) sentence-score threshold for Provence, (c) history window size k and similarity threshold for history pruning, and (d) generator model (Mistral 7B vs. Llama 3.1 8B).

6. Expected Challenges

• LLM judge reliability: Small judge models (≤7B) may exhibit positional bias or inconsistent scoring. Mitigation: multi-sample majority voting and calibration against human labels.
• Aggressive pruning vs. faithfulness trade-off: Token-level compression at high ratios (>5×) may discard evidence needed for faithful generation. This tension will be systematically analyzed via the compression–faithfulness Pareto curve.
• Provence license: The naver/provence-reranker-debertav3-v1 checkpoint is CC-BY-NC-4.0. For reproducibility, we will also train a lightweight DeBERTa-base pruning head on MS MARCO following the published procedure.
• Compute constraints: 7B model inference on GPU is feasible (RTX 3090/A100); however, QReCC's 54M-passage collection requires careful indexing, retrieval caching, batching, and parallelization.
• History pruning evaluation: QReCC provides real conversational histories, so the final system should evaluate on QReCC rather than relying on synthetic two-turn histories.
 

CENG 467 Spring 2026  ·  Custom Project Proposal  ·  İzmir Institute of Technology
