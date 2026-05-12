# LNN Model-Based History Pruner — Technical Review, Research Notes, and Improvement Plan

## 0. Goal

Bu doküman, mevcut **LNN sentence-level history pruner** kodunu teknik olarak review eder ve akademik literatürdeki context pruning / dialogue pruning / Liquid Neural Network fikirlerine göre nasıl daha güçlü bir modele dönüştürülebileceğini özetler.

Mevcut modelin core fikri doğru:

> Conversation history pruning’i sadece static similarity problemi olarak değil, zamansal olarak akan konuşma geçmişinde her sentence/turn için relevance score tahmin eden bir sequence labeling problemi olarak ele almak.

Ancak mevcut hali daha çok minimal bir V1. Daha güçlü ve akademik olarak savunulabilir hale getirmek için model şu yöne evrilmeli:

> **Query-aware, sequence-labeling based, structure-preserving, budget-aware Liquid history pruner.**

Önerilen final isim:

> **Q-LiSP: Query-Aware Liquid Span Pruner**

Alternatif isim:

> **Query-Aware Liquid History Pruner**

---

# 1. Mevcut modelin özeti

Mevcut kodda iki ana sınıf var:

```text
LNNSentenceModel
LNNSentencePruner
```

Modelin çalışması şu şekilde:

```text
passages
  ↓
sentence split
  ↓
sentence embedding + query embedding + turn feature
  ↓
projection layer
  ↓
LTC cell sequential scoring
  ↓
sentence importance scores
  ↓
threshold + max_tokens ile seçim
  ↓
tek passage olarak pruned history
```

Mevcut mimari:

```text
input = sentence_embedding ⊕ query_embedding ⊕ turn_feature
      ↓
Linear projection
      ↓
LayerNorm + GELU
      ↓
LTC Cell
      ↓
MLP scorer
      ↓
Sigmoid importance score
```

Bu yapı gerçekten **model-based sentence-level history pruner** sayılır. Çünkü:

* Her sentence için ayrı relevance score üretiyor.
* Current query embedding’ini modele veriyor.
* Sentence’ları sequential olarak LTC cell ile işliyor.
* Token budget altında en yüksek skorlu sentence’ları seçiyor.
* Hidden state taşıyarak conversation dynamics yakalamaya çalışıyor.

---

# 2. Mevcut modelin güçlü yanları

## 2.1. Model-based olması

Baseline pruner’lar genelde şunlara dayanır:

```text
recency-only
similarity-only
BM25 / TF-IDF
fixed top-k retrieval
```

Senin model ise öğrenilebilir bir scoring function kullanıyor:

```text
history sentences + query → learned keep/drop scores
```

Bu akademik olarak daha güçlü bir problem formülasyonu.

## 2.2. Sequence labeling fikrine yakın olması

Her sentence için skor üretmen doğru:

```text
sentence_i → keep/drop score
```

Bu, context pruning literatüründe kullanılan sequence labeling yaklaşımıyla uyumlu.

## 2.3. LNN/LTC kullanımı

LTC cell, sentence’ları sırayla işlerken hidden state tutuyor. Bu sayede model şunu öğrenebilir:

```text
Bir sentence tek başına önemli görünmeyebilir,
ama önceki konuşma akışı içinde önemli olabilir.
```

Örneğin:

```text
m1: “Option B olarak LNN-based pruning yapalım.”
m2: “Tamam bunu yapalım.”
```

m2 tek başına anlamsızdır ama m1 ile birlikte önemlidir. Sequential model bunu static similarity’den daha iyi yakalayabilir.

## 2.4. Token budget’a saygı göstermesi

Pruner, `max_tokens` limitini gözetiyor. Bu önemli çünkü history pruning’in amacı sadece relevance değil, aynı zamanda context budget yönetimi.

---

# 3. Mevcut modeldeki ana eksikler

## 3.1. Explicit query similarity yok

Şu an input şu:

```python
input_dim = emb_dim * 2 + 1
# sent_emb + query_emb + turn_feat
```

Bu modelin query-sentence ilişkisini kendisinin öğrenmesini bekliyor. Daha iyi yaklaşım, modele explicit query interaction sinyalleri vermek:

```text
cosine_similarity(sentence, query)
sentence_embedding * query_embedding
abs(sentence_embedding - query_embedding)
```

Sadece concat etmek zayıf kalabilir.

---

## 3.2. Turn feature zayıf

Şu an kullanılan feature:

```python
turn_feat = self._turn_index / 10.0
```

Bu, aynı prune çağrısındaki tüm sentence’lara aynı değeri verir. Yani model sentence’ların kendi iç sırasını öğrenemez.

Daha iyi feature’lar:

```text
position_in_history
recency
position_in_passage
sentence_index_in_turn
length_norm
```

Özellikle history pruning için recency güçlü bir sinyaldir.

---

## 3.3. Passage metadata kayboluyor

Mevcut kodda sentence’lar string olarak toplanıyor:

```python
all_sents = []
for p in passages:
    for sent in split_sentences(p["passage"]):
        all_sents.append(sent)
```

Bu sırada şu bilgiler kayboluyor:

```text
sentence hangi passage’dan geldi?
passage title neydi?
retriever score neydi?
retriever rank neydi?
sentence passage içinde kaçıncıydı?
```

Context pruning ve reranking literatürüne göre retrieval score/rank gibi sinyaller önemli olabilir.

---

## 3.4. Scorer sadece LTC output kullanıyor

Mevcut scorer:

```python
score = scorer(output)
```

Bu bazen local sentence bilgisinin kaybolmasına neden olabilir. Daha iyi:

```python
score_input = concat(ltc_output, projected_input)
score = scorer(score_input)
```

Bu residual scorer gibi çalışır.

---

## 3.5. Sigmoid model içinde uygulanıyor

Mevcut model inference için uygun ama training açısından daha iyi pratik:

```python
model → logits
training → BCEWithLogitsLoss
inference → sigmoid(logits)
```

`BCEWithLogitsLoss`, sigmoid + BCE’den daha numerically stable’dır.

---

## 3.6. Top sentence selection coherence kırabilir

Şu an seçim mantığı:

```text
score’a göre sentence’ları sırala
token budget dolana kadar al
sonra original order’a dön
```

Bu çalışır ama coherent context üretmeyebilir. Örneğin:

```text
m2 sentence 1
m7 sentence 3
m14 sentence 2
```

LLM’e verildiğinde context kopuk olabilir.

Daha iyi yaklaşım:

```text
sentence score → smoothing → coherent span oluşturma → budget-aware span selection
```

---

# 4. Akademik araştırmalardan çıkan ana fikirler

## 4.1. Provence: context pruning = sequence labeling + reranking

**Paper:** Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation
**Authors:** Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant
**Venue:** ICLR 2025
**Core idea:** Context pruning, context içindeki parçalar için binary mask / sequence labeling problemi olarak ele alınabilir. Ayrıca pruning ve reranking aynı model içinde birleştirilebilir.

Bu paper’dan alınacak fikir:

```text
Pruner sadece sentence scorer olmasın;
retrieval rank/score ve reranking sinyalleriyle birlikte çalışsın.
```

Senin modele etkisi:

```text
input feature’larına retriever_score, retriever_rank, passage_rank ekle.
```

Model framing:

> We formulate history pruning as a query-conditioned sequence labeling problem, where each sentence receives a keep/drop relevance score.

---

## 4.2. DyCP: long-form dialogue için coherent span seçimi

**Paper:** Dynamic Context Pruning for Long-Form Dialogue with LLMs
**Authors:** N. Choi et al.
**Year:** 2026
**Core idea:** Long dialogue history’de tek tek mesaj seçmek yerine, current query’ye göre sustained relevance gösteren contiguous dialogue spans seçmek daha iyi coherence sağlar.

Bu paper’dan alınacak fikir:

```text
Individual sentence top-k yerine contiguous span selection.
```

Senin modele etkisi:

```text
LNN sentence score üretsin,
selector ise coherent spans seçsin.
```

Önerilen selector:

```text
1. sentence scores üret
2. neighbor smoothing uygula
3. threshold üstü ardışık sentence’ları span olarak grupla
4. span_score hesapla
5. token budget altında en iyi span’leri seç
6. original order’da döndür
```

---

## 4.3. AttentionRAG: query focus / query-context alignment

**Paper:** AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation
**Authors:** Yixiong Fang, Tianran Sun, Yuling Shi, Xiaodong Gu
**Year:** 2025
**Core idea:** RAG context pruning’de query-context alignment çok önemli. Query semantic focus daha net çıkarılırsa context pruning daha başarılı olur.

Bu paper’dan alınacak fikir:

```text
Query embedding’i sadece concat etmek yetmez;
query-aware interaction features kullanılmalı.
```

Senin modele etkisi:

```python
features = [
    sent_emb,
    query_emb,
    sent_emb * query_emb,
    abs(sent_emb - query_emb),
    cosine_similarity,
]
```

Bu, bi-encoder embedding üstünde basit bir cross-interaction sağlar.

---

## 4.4. Liquid Time-Constant Networks: temporal relevance dynamics

**Paper:** Liquid Time-Constant Networks
**Authors:** Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu
**Venue:** AAAI 2021
**Core idea:** LTC networks, continuous-time recurrent neural network ailesinden gelir. Hidden-state’e bağlı değişken time constants ile temporal dynamics modelleyebilir.

Bu paper’dan alınacak fikir:

```text
Dialogue history zamansal bir akıştır.
Sentence relevance bağımsız değildir; konuşma ilerledikçe değişir.
```

Senin model framing’in:

> We investigate whether Liquid Time-Constant recurrent dynamics can improve temporal relevance modeling in dialogue history pruning compared to static similarity and recency baselines.

Dikkat edilmesi gereken nokta:

```text
“LNN kesin daha iyidir” demek riskli.
```

Daha akademik iddia:

```text
LNN/LTC is a hypothesis for modeling temporal relevance dynamics.
```

---

# 5. Önerilen yeni model: Query-Aware Liquid Span Pruner

## 5.1. Final architecture

```text
Input:
  history H = {u1, a1, ..., ut}
  current query q
  token budget B

Pipeline:
  1. Split history into sentences or turns.
  2. Encode each sentence and query using a frozen sentence encoder.
  3. Build query-aware interaction features.
  4. Process sequence with LNN/LTC or BiLTC.
  5. Predict sentence relevance logits.
  6. Apply neighbor smoothing.
  7. Build coherent candidate spans.
  8. Select spans under budget.
  9. Return selected spans in original order.
```

---

## 5.2. Feature vector

Mevcut:

```python
features_i = concat(sent_emb, query_emb, turn_feat)
```

Önerilen:

```python
features_i = concat(
    sent_emb,
    query_emb,
    sent_emb * query_emb,
    abs(sent_emb - query_emb),
    cosine_similarity(sent_emb, query_emb),
    position_in_history,
    recency,
    length_norm,
    turn_index,
    passage_rank,
    retriever_score,
    role_feature,
    rule_features,
)
```

Eğer embedding dim 384 ise:

```text
sent_emb: 384
query_emb: 384
sent_emb * query_emb: 384
abs(sent_emb - query_emb): 384
scalar features: ~8-12

input_dim ≈ 1544-1548
```

Bu büyük değil çünkü projection layer 128/256’ya indiriyor.

---

## 5.3. Rule features

Rule features modelin özellikle project/debug konuşmalarında daha iyi çalışmasını sağlar.

Önerilen binary features:

```text
contains_code
contains_error
contains_decision
contains_dataset_name
contains_model_name
contains_number
is_user_message
is_question
contains_file_reference
contains_requirement_keyword
```

Örnek decision keywords:

```text
"let's use"
"we decided"
"bunu yapalım"
"tamam"
"option B"
"QuAC kullanalım"
"QReCC'e geçelim"
```

---

## 5.4. Model body: V1.5 ve V2

### V1.5 — düşük riskli geliştirme

```text
Query-aware features
  ↓
Projection
  ↓
Single-direction LTC
  ↓
Residual scorer
  ↓
Sentence logits
```

Bu mevcut koddan az değişiklikle yapılabilir.

### V2 — daha güçlü model

```text
Query-aware features
  ↓
Projection
  ↓
Forward LTC + Backward LTC
  ↓
concat(forward_h, backward_h, projected_input)
  ↓
Residual scorer
  ↓
Sentence logits
```

V2 daha iyi olabilir çünkü sentence relevance sadece önceki değil sonraki sentence’lardan da etkilenebilir.

Fakat online streaming history pruning yapacaksan single-direction daha doğal. Offline evaluation veya batch pruning için BiLTC daha güçlüdür.

---

# 6. Proposed PyTorch-level model design

## 6.1. Model logits döndürmeli

Training için model sigmoid değil logits döndürmeli:

```python
logit = self.scorer(score_input)
```

Inference:

```python
scores = torch.sigmoid(logits)
```

Training:

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 6.2. Residual scorer

Mevcut scorer:

```python
score = self.scorer(output)
```

Önerilen scorer:

```python
score_input = torch.cat([output, x_i], dim=-1)
logit = self.scorer(score_input)
```

Scorer:

```python
self.scorer = nn.Sequential(
    nn.Linear(output_units + proj_dim, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
)
```

---

## 6.3. V1.5 model pseudocode

```python
class QueryAwareLTCPrunerModel(nn.Module):
    def __init__(self, input_dim, proj_dim=128, hidden_units=64, output_units=32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        from ncps.wirings import AutoNCP
        from ncps.torch import LTCCell

        wiring = AutoNCP(units=hidden_units, output_size=output_units)
        self.ltc_cell = LTCCell(wiring, in_features=proj_dim)
        self.state_size = self.ltc_cell.state_size

        self.scorer = nn.Sequential(
            nn.Linear(output_units + proj_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features, hx=None):
        projected = self.projection(features)

        if hx is None:
            hx = torch.zeros(1, self.state_size, device=features.device)

        logits = []
        for i in range(projected.size(0)):
            x_i = projected[i:i+1]
            output, hx = self.ltc_cell(x_i, hx)
            score_input = torch.cat([output, x_i], dim=-1)
            logit = self.scorer(score_input)
            logits.append(logit.squeeze(-1))

        logits = torch.cat(logits, dim=0)
        return logits, hx
```

---

# 7. Better selector: coherent span selection

## 7.1. Neden gerekli?

Top sentence selection context coherence’i bozabilir.

Örnek kötü output:

```text
Sentence 3 from passage 1
Sentence 14 from passage 5
Sentence 2 from passage 9
```

LLM için bu kopuk context oluşturur.

Daha iyi output:

```text
Span 1: sentences 3-5
Span 2: sentences 12-15
```

---

## 7.2. Neighbor smoothing

Sentence scores:

```python
scores = sigmoid(logits)
```

Smoothing:

```python
smoothed[i] = 0.2 * scores[i-1] + 0.6 * scores[i] + 0.2 * scores[i+1]
```

Boundary cases için:

```python
left = scores[i-1] if i > 0 else scores[i]
right = scores[i+1] if i < n-1 else scores[i]
smoothed[i] = 0.2 * left + 0.6 * scores[i] + 0.2 * right
```

Bu, evidence chain kırılmasını azaltır.

---

## 7.3. Candidate span building

```python
def build_spans(scores, threshold):
    spans = []
    start = None

    for i, score in enumerate(scores):
        if score >= threshold and start is None:
            start = i
        elif score < threshold and start is not None:
            spans.append((start, i - 1))
            start = None

    if start is not None:
        spans.append((start, len(scores) - 1))

    return spans
```

---

## 7.4. Span scoring

```python
span_score = mean(scores[start:end+1]) + 0.1 * max(scores[start:end+1])
```

Token-normalized:

```python
span_value = span_score / sqrt(num_tokens)
```

Bu uzun ama düşük yoğunluklu span’lerin her şeyi kaplamasını engeller.

---

## 7.5. Budget-aware span selection

```python
candidate_spans = sorted(spans, key=lambda s: s.value, reverse=True)
selected = []
token_count = 0

for span in candidate_spans:
    if token_count + span.num_tokens <= max_tokens:
        selected.append(span)
        token_count += span.num_tokens

selected.sort(key=lambda s: s.start)
```

Sonra original order’da text döndür.

---

# 8. Training strategy

## 8.1. Label üretimi

### HotpotQA

HotpotQA’da supporting facts olduğu için label üretimi daha kolay:

```text
sentence supporting fact ise label = 1
otherwise label = 0
```

### QuAC / QReCC

Bu datasetlerde label daha zor olabilir. Pseudo-label stratejisi:

```python
label_i = 1 if (
    answer_overlap(sentence_i, gold_answer) > threshold
    or cosine(sentence_i, gold_answer) > threshold
    or LLM_judge_says_relevant(sentence_i, query, answer)
) else 0
```

Başlangıçta LLM judge şart değil. Overlap + embedding similarity yeterli MVP olur.

---

## 8.2. Soft neighbor labels

Evidence chain korumak için positive sentence komşularına soft label verilebilir:

```text
gold sentence: 1.0
neighbor sentence: 0.4
others: 0.0
```

Bu, modelin tek cümleye aşırı keskin odaklanmasını engeller.

---

## 8.3. Loss function

Sadece BCE kullanabilirsin ama daha güçlü loss:

```python
loss = bce_loss + 0.2 * ranking_loss + 0.1 * smoothness_loss
```

### BCE loss

```python
bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
```

Class imbalance için `pos_weight` önemli çünkü çoğu sentence irrelevant olacaktır.

### Pairwise ranking loss

Amaç:

```text
positive sentence score > negative sentence score
```

```python
ranking_loss = mean(max(0, margin - score_pos + score_neg))
```

### Smoothness loss

Komşu sentence score’ları aşırı zıplamasın:

```python
smoothness_loss = mean((scores[1:] - scores[:-1]) ** 2)
```

Bu span coherence’i destekler.

---

# 9. Evaluation design

## 9.1. Baselines

Mutlaka karşılaştır:

```text
1. No pruning / full context
2. Recency-only pruning
3. Similarity-only pruning
4. BM25 / TF-IDF pruning
5. Existing sentence embedding top-k
6. LNN sentence pruner V1
7. Query-aware LNN V1.5
8. Query-aware Liquid Span Pruner V2
```

## 9.2. Metrics

Context pruning için iki grup metric kullan:

### Compression metrics

```text
compression ratio
input tokens before pruning
input tokens after pruning
latency
```

### Answer quality metrics

Dataset’e göre:

```text
Exact Match
F1
ROUGE-L
BERTScore
LLM-as-a-judge correctness
```

Asıl rapor cümlesi:

> We evaluate whether the pruner preserves answer quality under increasingly smaller context budgets.

Yani hedef:

```text
Daha az token ile full-context answer quality’ye yaklaşmak.
```

---

# 10. Concrete patch plan for current code

## Patch 1 — metadata-aware sentence collection

Mevcut:

```python
all_sents = []
for p in passages:
    for sent in split_sentences(p["passage"]):
        all_sents.append(sent)
```

Önerilen:

```python
all_items = []
for p_idx, p in enumerate(passages):
    sentences = split_sentences(p["passage"])
    for s_idx, sent in enumerate(sentences):
        all_items.append({
            "text": sent,
            "passage_idx": p_idx,
            "sent_idx": s_idx,
            "title": p.get("title"),
            "retriever_score": float(p.get("score", 0.0) or 0.0),
            "rank": float(p.get("rank", p_idx + 1) or p_idx + 1),
            "role": p.get("role", None),
        })
```

---

## Patch 2 — query-aware feature builder

```python
texts = [item["text"] for item in all_items]
sent_embs = self.encoder.encode(texts)
query_emb = self.encoder.encode([query])[0]
q_broadcast = np.tile(query_emb, (len(texts), 1))

cos = np.sum(sent_embs * q_broadcast, axis=1, keepdims=True) / (
    np.linalg.norm(sent_embs, axis=1, keepdims=True) *
    np.linalg.norm(q_broadcast, axis=1, keepdims=True) + 1e-8
)

prod = sent_embs * q_broadcast
abs_diff = np.abs(sent_embs - q_broadcast)

n = len(texts)
pos = np.arange(n).reshape(-1, 1) / max(n - 1, 1)
recency = 1.0 - pos
lengths = np.array([self._count_tokens(t) for t in texts], dtype=np.float32).reshape(-1, 1)
length_norm = lengths / max(float(lengths.max()), 1.0)

rank = np.array([item["rank"] for item in all_items], dtype=np.float32).reshape(-1, 1)
rank_norm = 1.0 / np.maximum(rank, 1.0)

retriever_score = np.array(
    [item["retriever_score"] for item in all_items],
    dtype=np.float32
).reshape(-1, 1)

features = np.concatenate([
    sent_embs,
    q_broadcast,
    prod,
    abs_diff,
    cos,
    pos,
    recency,
    length_norm,
    rank_norm,
    retriever_score,
], axis=1)
```

---

## Patch 3 — model input_dim update

Eski:

```python
input_dim = emb_dim * 2 + 1
```

Yeni:

```python
scalar_dim = 5  # cos, pos, recency, length_norm, rank_norm/retriever_score etc.
input_dim = emb_dim * 4 + scalar_dim
```

Not: Kaç scalar eklediğine göre güncelle.

---

## Patch 4 — logits + residual scorer

Model içinde:

```python
score_input = torch.cat([output, x_i], dim=-1)
logit = self.scorer(score_input)
```

Return:

```python
return logits, hx
```

Inference:

```python
logits, self._hidden_state = self._model(features_t, self._hidden_state)
scores = torch.sigmoid(logits).cpu().numpy()
```

---

## Patch 5 — span selector

Eski:

```python
indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
kept = []
...
```

Yeni:

```python
scores = smooth_scores(scores)
spans = build_spans(scores, threshold=self.threshold)
selected_spans = select_spans_under_budget(spans, scores, all_items, max_tokens)
retained_text = render_spans(selected_spans, all_items)
```

---

## Patch 6 — debug output

Return içine metadata ekle:

```python
return [{
    "passage": retained_text,
    "title": None,
    "score": float(np.mean([s["score"] for s in selected_debug])) if selected_debug else 0.0,
    "rank": 1,
    "pruned": True,
    "metadata": {
        "kept_sentences": selected_debug,
        "input_tokens": self._last_input_tokens,
        "output_tokens": self._last_output_tokens,
        "compression_ratio": self._last_output_tokens / max(self._last_input_tokens, 1),
    }
}]
```

Bu qualitative analysis için çok değerli olur.

---

# 11. Recommended versioning

## V1 — current model

```text
Sentence embedding + query embedding + turn feature
Single LTC
Sigmoid scorer
Top-score sentence selection
```

Status:

```text
Good MVP, but minimal.
```

## V1.5 — recommended immediate upgrade

```text
Query-aware interaction features
Position/recency/length features
Residual scorer
Logits output
Metadata-aware output
```

Status:

```text
Best short-term improvement.
```

## V2 — final project model

```text
Query-aware features
BiLTC or single LTC
Neighbor smoothing
Coherent span selector
Budget-aware selection
BCE + ranking + smoothness loss
```

Status:

```text
Best academic contribution.
```

## V3 — research-level extension

```text
Differentiable top-k / reinforcement learning
Answer-quality reward
LLM judge feedback
Adaptive compression ratio
```

Status:

```text
Too risky for immediate project unless extra time exists.
```

---

# 12. Suggested report framing

## Problem statement

> Long conversational histories contain redundant and irrelevant information, increasing inference cost and distracting the downstream language model. We address this by learning a query-conditioned history pruner that selects the most relevant parts of the dialogue under a token budget.

## Method statement

> We formulate history pruning as a sequence labeling problem. Each sentence in the dialogue history receives a keep/drop relevance score conditioned on the current query. Instead of scoring sentences independently, we use a Liquid Time-Constant recurrent model to capture temporal relevance dynamics across the conversation.

## Improved model statement

> To preserve coherence, we do not directly select isolated top-scoring sentences. Instead, we smooth sentence-level scores and construct contiguous candidate spans, selecting the highest-value spans under the token budget.

## LNN justification

> Dialogue history is a temporally ordered stream where the relevance of a sentence depends on previous and subsequent turns. Liquid recurrent models provide a natural way to investigate whether continuous-time-inspired dynamics can improve temporal relevance modeling over static similarity-based pruning.

## Careful claim

Avoid:

```text
LNN is better than all other pruners.
```

Use:

```text
We investigate whether LNN-based temporal relevance modeling improves pruning quality compared with recency and similarity baselines.
```

---

# 13. Final recommendation

Mevcut modeli çöpe atma. Doğru yönde ilerliyor.

Ama final project için sadece şu model zayıf kalabilir:

```text
sent_emb + query_emb + LTC + top-k selection
```

Daha güçlü final contribution şu olmalı:

```text
Query-aware Liquid Span Pruner
```

Yani:

```text
LNN sentence scorer
+ query interaction features
+ metadata-aware retrieval signals
+ residual scoring head
+ coherent span selection
+ budget-aware pruning
```

Bunun akademik katkısı daha net:

> Static similarity-based pruning ignores temporal dialogue dynamics and may select incoherent isolated sentences. We propose a query-aware Liquid history pruner that models temporal relevance over the dialogue and selects coherent spans under a token budget.

---

# 14. References

1. Chirkova, N., Formal, T., Nikoulina, V., & Clinchant, S. (2025). **Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation.** ICLR 2025. arXiv:2501.16214.

2. Choi, N. et al. (2026). **Dynamic Context Pruning for Long-Form Dialogue with LLMs.** arXiv:2601.07994.

3. Fang, Y., Sun, T., Shi, Y., & Gu, X. (2025). **AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation.** arXiv:2503.10720.

4. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). **Liquid Time-Constant Networks.** Proceedings of the AAAI Conference on Artificial Intelligence, 35(9), 7657–7666.
