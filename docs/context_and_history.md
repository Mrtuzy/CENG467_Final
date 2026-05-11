## 1. Retrieval-side pruning

RAG’de LLM’e girmeden önce yapılır.

**Chunk filtering / top-k retrieval**
Sadece en alakalı chunk’ları seçmek.

**Reranking**
İlk retrieval geniş yapılır, sonra cross-encoder / LLM / embedding reranker ile en iyi parçalar seçilir.

**MMR / diversity-aware retrieval**
Birbirinin aynısı chunk’ları azaltır; hem relevant hem diverse context seçer.

**Query-focused summarization**
Dokümanı genel özetlemek yerine, kullanıcının sorusuna göre özetler.

**Hierarchical retrieval**
Önce doküman/section seçilir, sonra o section içinden paragraph/chunk seçilir.

**Parent-child retrieval**
Küçük chunk ile retrieve edilir ama modele daha büyük parent block verilir; bazen tersine, parent bulunup child’lar prune edilir.

**Metadata filtering**
Tarih, source, author, topic, permission, document type gibi filtrelerle context azaltılır.

**Deduplication / near-duplicate removal**
Benzer chunk’ları cosine similarity veya MinHash gibi yöntemlerle temizler.

**Clustering-based context compression**
Benzer chunk’ları cluster’layıp her cluster’dan temsilci veya özet seçer. RAG için dynamic clustering-based document compression gibi yeni çalışmalar da var. ([EMNLP 2025][1])

---

## 2. Prompt-level compression

LLM’e girecek prompt daha oluşturulurken küçültülür.

**Prompt compression**
LLM / küçük model / saliency skorları ile gereksiz token’ları atar.

**Instruction pruning**
Sistem promptu, tool açıklamaları, policy textleri, format yönergeleri gibi sabit parçaların gereksiz detaylarını azaltır.

**Few-shot example selection**
Tüm örnekleri koymak yerine sadece task’a en yakın birkaç örnek seçilir.

**Example compression**
Few-shot örneklerin input/output kısmı kısaltılır.

**Schema compression**
Tool schema, JSON schema, API doc gibi uzun yapılar minimal forma indirilir.

**Context distillation**
Uzun geçmiş veya doküman, daha kısa bir “state” ya da “working memory” temsiline dönüştürülür.

**Conversation summarization**
Chat history’nin eski kısmı özetlenir.

**Recursive summarization**
Uzun geçmiş parça parça özetlenir, sonra özetlerin özeti alınır.

**Map-reduce summarization**
Her doküman/chunk ayrı özetlenir, sonra final synthesis yapılır.

**Selective summarization**
Her şeyi özetlemek yerine sadece kararlar, constraints, entities, açık tasklar tutulur.

---

## 3. Attention / saliency based pruning

Modelin veya yardımcı bir modelin “hangi token önemli?” sinyalini kullanır.

**Attention-guided compression**
Attention skorlarına göre context token’ları tutulur veya atılır. EMNLP 2025 listesinde “Attention-Guided Adaptive Context Compression for RAG” gibi çalışmalar bu hatta giriyor. ([EMNLP 2025][1])

**Gradient / attribution-based pruning**
Token önemini gradient, integrated gradients veya attribution skorlarıyla ölçüp prune eder.

**Perplexity / loss-based pruning**
Modelin prediction’ına az katkı yapan token’lar çıkarılır.

**Entity-aware pruning**
Named entity, date, number, code symbol, citation gibi kritik token’lar korunur.

**Position-aware pruning**
Baş/son kısımlar, section title’lar, question-near spans gibi pozisyonlar önceliklendirilir.

**Recency-aware pruning**
Chat sistemlerinde en yeni mesajlar daha yüksek öncelik alır.

---

## 4. KV cache pruning / eviction

Generation sırasında KV cache’i küçültür. Senin dediğin **KV cache eviction** burada.

**Static KV cache selection**
Prefill sırasında önemli token’lar seçilir, decoding boyunca bu seçim sabit kalır.

**Dynamic KV cache selection**
Decoding sırasında cache sürekli güncellenir; seçilmeyen token’lar evict edilir veya başka belleğe offload edilir. KV cache acceleration survey’lerinde bu ayrım açık şekilde kullanılıyor. ([arXiv][2])

**Heavy hitter token retention**
Attention’da sürekli yüksek skor alan token’lar tutulur.

**Streaming / sliding-window cache**
Son N token tutulur, eski token’lar atılır; StreamingLLM tarzı yaklaşımlar bu ailede.

**Sink token preservation**
Bazı ilk token’lar attention sink gibi davrandığı için tamamen atılmaz, korunur.

**Layer-wise KV pruning**
Her layer’da aynı token’ları tutmak yerine layer’a göre farklı pruning yapılır.

**Head-wise KV pruning**
Her attention head için farklı token importance hesaplanır.

**Block-level KV eviction**
Tek tek token yerine bloklar halinde cache atılır; daha sistem dostudur.

**Learned KV retention**
Token tutma/atma kararı öğrenilmiş gate veya skor modeliyle yapılır. Yeni çalışmalarda inference sırasında token skorlayıp eviction yapan retention gate yaklaşımları var. ([arXiv][3])

**KV offloading**
Token’ı tamamen silmek yerine GPU’dan CPU memory’ye veya daha yavaş storage’a taşır.

**Tiered KV memory / memory hierarchy**
GPU cache, CPU memory, disk/persistent memory gibi katmanlı yapı. 2026 tarihli “demand paging for LLM systems” çalışması bunu L1 eviction, L2 pinning, L3 conversation compaction gibi katmanlarla ele alıyor. ([arXiv][4])

---

## 5. KV cache compression

Burada token atmak yerine cache temsilini küçültürsün.

**KV quantization**
KV cache FP16 yerine INT8/INT4 gibi daha küçük precision’da tutulur.

**Low-rank KV compression**
KV tensorleri low-rank approximation ile sıkıştırılır.

**KV merging**
Benzer KV vektörleri birleştirilir. Senin dediğin **token merging** bununla ilişkili.

**Feature merging**
Token’ı değil, KV feature representation’larını merge eder.

**Residual / compensated merging**
Merge ederken kaybolan bilgiyi residual correction ile telafi etmeye çalışır. ZeroMerge gibi çalışmalar token pruning ve feature merging’in bilgi kaybı sorununu azaltmaya odaklanıyor. ([arXiv][5])

**Attention-score guided KV compression**
Önemli KV pair’ler attention skorlarına göre tutulur. TableKV, büyük tabloları in-context işlerken attention skorlarıyla önemli KV pair’leri korumayı öneriyor. ([ACL Anthology][6])

---

## 6. Token merging / representation merging

Input veya internal representation seviyesinde yapılabilir.

**Token merging**
Benzer token representation’ları birleştirilir.

**Semantic span merging**
Token token değil, phrase/span seviyesinde merge yapılır.

**Redundant sentence merging**
Aynı bilgiyi veren cümleler tek cümle/temsil haline getirilir.

**Cluster representative selection**
Birbirine benzeyen parçalar cluster’lanır, sadece merkez/temsilci seçilir.

**Latent compression**
Text’i tekrar text olarak değil, latent vector/state olarak saklar. Senin Text-JEPA/RAG fikrine yakın yer burası.

---

## 7. Agentic paging / memory paging

Senin dediğin **agentic paging** daha “OS gibi memory management” tarafı.

**Agentic paging**
Agent gerektiğinde eski context sayfalarını çağırır; tüm geçmiş prompt’a konmaz.

**Demand paging**
Model/agent ihtiyaç duyunca memory page fetch eder. 2026’daki demand paging çalışması LLM sistemleri için L1-L3 memory hierarchy fikrini tartışıyor. ([arXiv][4])

**Scratchpad compaction**
Agent’ın ara düşünce, tool sonucu, plan ve observation geçmişi düzenli olarak compact edilir.

**Working memory vs long-term memory split**
Kısa vadeli aktif context ayrı, uzun vadeli storage ayrı tutulur.

**Episodic memory retrieval**
Eski konuşma/event’ler sadece gerektiğinde retrieve edilir.

**Semantic memory retrieval**
Kullanıcının kalıcı tercihleri, facts, proje bilgileri gibi şeyler ayrı memory’den çekilir.

**Tool-result paging**
Uzun tool outputs context’e komple koyulmaz; gerektiğinde belirli satır/blok geri çağrılır.

---

## 8. Structural pruning

Context’in yapısına göre budama.

**HTML / DOM pruning**
Web sayfasında nav, footer, sidebar, ads, boilerplate temizlenir.

**Code context pruning**
Sadece ilgili function/class/import/test dosyaları context’e alınır.

**AST-based pruning**
Kodda raw text yerine AST üzerinden ilgili node’lar seçilir.

**Document section pruning**
Başlık hiyerarşisine göre sadece ilgili section’lar tutulur.

**Table pruning**
Tabloda sadece ilgili satır/sütun/cell’ler seçilir.

**Citation/source pruning**
RAG’de sadece cevabı destekleyen kaynaklar tutulur, zayıf kaynaklar atılır.

---

## 9. Compression by transformation

Bilgi korunur ama format değişir.

**Natural language → structured state**
Uzun text, JSON state’e çevrilir.

Örnek:

```json
{
  "goal": "...",
  "constraints": ["..."],
  "decisions": ["..."],
  "open_questions": ["..."]
}
```

**Triples / knowledge graph extraction**
Text, subject-predicate-object triple’larına dönüştürülür.

**Entity-state tracking**
Sadece entity’lerin son durumları tutulur.

**Timeline compression**
Olaylar kronolojik kısa timeline’a çevrilir.

**Decision log compression**
Sadece alınan kararlar ve gerekçeler tutulur.

**Task-state compression**
Agentic workflow’da sadece yapılacaklar, tamamlananlar, blockers tutulur.

---

## 10. Model-routing based pruning

Context’i tek modele vermek yerine farklı modellerle azaltırsın.

**Small model pre-filtering**
Küçük model önce gereksiz context’i atar.

**LLM judge pruning**
Bir LLM, “bu chunk cevap için gerekli mi?” diye karar verir.

**Classifier-based relevance pruning**
Binary classifier: relevant / irrelevant.

**Embedding + LLM hybrid pruning**
Embedding ile hızlı shortlist, LLM ile son seçim.

**Budget-aware context assembly**
Token budget’a göre hangi parçaların gireceği optimize edilir.

---

## 11. Retrieval-time compression indexes

Index’in kendisi compression-aware olur.

**Summary index**
Her chunk’ın kısa özeti indexlenir.

**Multi-vector retrieval**
Dokümanın farklı aspect’leri için birden fazla vector tutulur; sadece ilgili aspect çekilir.

**Late interaction retrieval**
ColBERT tarzı token-level matching ile daha fine-grained relevance yapılır.

**Hierarchical memory index**
Document → section → paragraph → sentence şeklinde ağaç yapı.

**Forward-looking retrieval**
Senin JEPA fikrine yakın: sadece mevcut query’ye benzeyen chunk değil, “cevabı destekleyecek sonraki semantic state”e benzeyen chunk çekilir.

