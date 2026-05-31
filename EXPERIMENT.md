# Entropy-Driven CfC Context Pruning: Experiment Design

## 1. Introduction
Large Language Models (LLMs) struggle with long context windows due to computational overhead and the "lost in the middle" phenomenon. In multi-turn conversational datasets like QReCC, not all past utterances are equally important for answering the final question.

This experiment investigates using a Continuous-time Neural Network (specifically, Closed-form Continuous-time networks or CfCs) to dynamically prune conversational context.

## 2. Theoretical Framework

### 2.1 The Concept of Cognitive Parallelism
Human cognition processes language not just word-by-word uniformly, but by allocating more processing time to highly "surprising" or entropic information. We map this concept to CfCs by utilizing their Liquid Time-Constant (LTC) mechanism.

### 2.2 Time-Entropy Mapping
Let $u_i$ be an utterance. We calculate its surprisal $S(u_i)$ using a proxy model (DistilGPT-2):
$$ S(u_i) = - \frac{1}{N} \sum_{j=1}^N \log P(w_j | w_{<j}) $$

This surprisal is mapped to a continuous time step $\Delta t_i$ for the CfC ODE solver:
$$ \Delta t_i = \Delta t_{min} + \beta \cdot S(u_i) $$
Where $\Delta t_{min} = 0.1$ and $\beta = 1.0$ (hyperparameters).

### 2.3 Knowledge Distillation for Ground Truth
To train the CfC network, we need ground truth importance scores ($p_{target}$) for each $u_i$. We obtain these using a "leave-one-out" approach with a powerful Teacher model (LLaMA-3.1-8B-Instruct).
If removing $u_i$ drastically changes the teacher's output (high KL-Divergence or high logit difference), $u_i$ is highly important.

### 2.4 The CfC Network
The CfC network takes sequence of embeddings $e_i$ (from SBERT) and continuous time-steps $\Delta t_i$, and outputs a probability $p_{pred} \in [0, 1]$ indicating the importance of $u_i$.
Loss function: MSE($p_{pred}, p_{target}$).

## 3. Evaluation
The framework will be evaluated against:
1. **Full Context:** Passing all history to the target LLM.
2. **Random Pruning:** Randomly removing utterances.
3. **Cosine Similarity Pruning:** Keeping utterances most similar to the final question (using SBERT).

Metrics:
* **Quality:** ROUGE-L, BERTScore
* **Efficiency:** Token count reduction, Time To First Token (TTFT) acceleration.
