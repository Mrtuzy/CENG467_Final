# CfC Context Pruning — Çalışma & Sunum Sheet'i
*(4–5 dk'lık sunum için hazırlanmış; CfC kısmı senin)*

---

## 0. Tek cümlelik özet
> Bir konuşmadaki **hangi geçmiş turn'lerin gerçekten gerekli** olduğunu öğrenen, sürekli-zamanlı
> küçük bir sinir ağı (CfC) eğitiyoruz; gereksiz turn'leri atıp (pruning) LLM'e daha kısa bağlam
> veriyoruz → **aynı cevap kalitesi, çok daha az token.**

---

## 1. Problem & Motivasyon (≈30 sn)

- Çok-turlu diyalogda (QReCC veri seti) LLM'e **tüm geçmişi** vermek pahalı: uzun bağlam = çok token
  = yüksek gecikme (TTFT) ve maliyet.
- Ama her turn eşit değerli değil — bazıları cevabı belirler, çoğu gürültü.
- **Amaç:** "context pruning" — cevabı bozmadan bağlamı kısalt.
- **Bizim katkımız:** Bunu **sürekli-zamanlı, hafif bir ağ (CfC)** ile yapan, **öğrenilmiş bir pruner**.
  Girdisi turn'lerin **anlamsal temsili (SBERT)**; ayrıca bir **bilgi-teorik sinyali (surprisal → zaman)**
  hipotez olarak test ediyoruz.

> ⚠️ **Dürüst bulgu (önemli sunum noktası):** "Surprisal, turn önemini öngörür mü?" hipotezini
> eğitimden önce ölçtük (Spearman ρ ≈ **0**, hatta hafif negatif). **Öngörmüyor.** Yani projenin
> gücü "entropy" sinyalinden değil, **CfC'nin öğrendiği içerik temsilinden** geliyor; Δt yalnızca
> zayıf bir zamansal prior. Bunu gizlemek yerine **test edip raporluyoruz** — bilimsel olgunluk.

---

## 2. Boru Hattı (Pipeline) — büyük resim (≈45 sn)

```
QReCC diyalog                e_i (SBERT 384-d)  ┐
   │  her turn u_i        ┌─► Δt_i (surprisal)  ┤──► [ CfC ]──► ŝ_i ∈[0,1]──► τ ile PRUNE
   │                      │                      ┘                              │
   └─ teacher (LLM) ──► imp_i (önem etiketi) ──────────► CfC'yi EĞİTİR ◄────────┘
                                                                                ▼
                                                            kısaltılmış bağlam → LLM → cevap
                                                            (ROUGE-L, token azalması ölç)
```

5 aşama:
1. **Veri:** QReCC → (bağlam turn'leri, soru, gold cevap).
2. **Girdi temsili:** her turn için **SBERT embedding** `e_i` + **surprisal'dan türetilen Δt_i**.
3. **Öğretmen etiketi:** bir LLM ile her turn'ün **gerçek önemi** `imp_i` (aşağıda).
4. **CfC eğitimi:** `(e_i, Δt_i) → ŝ_i` öğren, hedef `imp_i` (regresyon).
5. **Pruning + Değerlendirme:** `ŝ_i ≥ τ` olan turn'leri tut, LLM'e ver, kaliteyi/verimliliği ölç.

---

## 3. MATEMATİK (≈2 dk — sunumun kalbi)

### 3.1 Surprisal (bilgi sinyali)
Küçük bir dil modeli (DistilGPT-2) ile bir turn `u_i`'nin **ortalama negatif log-olasılığı**:

$$ S(u_i) = -\frac{1}{|u_i|}\sum_{t} \log P_\theta(w_t \mid w_{<t}) $$

Yüksek `S` = model bu turn'e "şaşırdı" = bilgi yoğun / öngörülemez içerik.

### 3.2 Surprisal → Δt eşlemesi (sürekli zamana çevir)
Surprisal'ı, CfC'nin "geçen zaman" girdisine çeviriyoruz. **Rank (sıra) normalizasyonu** kullanıyoruz
(aykırı surprisal'lara dayanıklı):

$$ \Delta t_i = \Delta t_{\min} + \beta \cdot \frac{\operatorname{rank}(S_i)}{N}, \qquad \Delta t_{\min}=0.05,\ \beta=1.0 $$

> Sezgi: şaşırtıcı turn → büyük Δt → CfC için "büyük bir zaman olayı" → hafıza durumunu daha çok günceller.

### 3.3 CfC nedir? (Closed-form Continuous-time)
Temeli **LTC (Liquid Time-Constant)** nöronu — bir adi diferansiyel denklem (ODE):

$$ \frac{dx}{dt} = -\Big[\tfrac{1}{\tau} + f(x, I)\Big] x + f(x, I)\, A $$

`x`: gizli durum, `I`: girdi, `τ`: zaman sabiti. **Sorun:** her adımda ODE çözmek yavaş.
**CfC çözümü:** Hasani vd. (2022) bu ODE'nin **kapalı-form (closed-form)** yaklaşık çözümünü verir,
ODE solver gerekmez:

$$ x(t) = \sigma\!\big(-[w_\tau + f(x,I)]\,t\big)\odot g(x,I) \;+\; \big[1-\sigma(-[w_\tau + f(x,I)]\,t)\big]\odot h(x,I) $$

Burada **`t` = bizim Δt_i'miz**. Yani geçen zaman, bir **kapı (gate)** içinde üstel olarak etki eder:
- Küçük Δt → durum az değişir (turn'ler birbirine "yakın").
- Büyük Δt → kapı açılır, yeni içerik durumu baskın günceller.

**Neden CfC?** (i) Düzensiz/sürekli zaman doğal şekilde işlenir — biz Δt'yi *surprisal'la* kuruyoruz;
(ii) çok küçük (64 nöron) ve hızlı; (iii) recurrent → turn'ler arası sıralı bağımlılığı yakalar.

### 3.4 Öğretmen etiketi — Cevap-koşullu Leave-One-Out ΔNLL
"Bir turn ne kadar önemli?" sorusuna **gerçek cevap üzerinden** yanıt veriyoruz. Bir LLM (Qwen2.5-1.5B)
ile gold cevap `a`'nın olabilirliğini ölçüyoruz:

$$ \text{NLL}_{\text{full}} = -\tfrac{1}{|a|}\sum \log P(a \mid C, q), \qquad
   \text{NLL}_{-i} = -\tfrac{1}{|a|}\sum \log P(a \mid C\setminus u_i, q) $$

$$ \boxed{\;\text{imp}_i = \max\!\big(0,\ \text{NLL}_{-i} - \text{NLL}_{\text{full}}\big)\;} $$

> Yani: turn `u_i`'yi attığımızda cevap **ne kadar zorlaşıyorsa** (NLL artıyorsa) o turn o kadar önemli.
> Sonra diyalog içinde [0,1] aralığına min-max normalize ediyoruz.
> (Eskiden full-vocab KL kullanılıyordu; bu hem yavaş hem dolaylıydı — ΔNLL doğrudan "cevap için önem"i ölçer.)

### 3.5 CfC öğrencisinin eğitimi
Her turn için CfC bir skor üretir: `ŝ_i = σ(W h_i) ∈ [0,1]`. Hedef `imp_i`. **Maskeli MSE** kaybı
(padding'i sayma):

$$ \mathcal{L} = \frac{\sum_i m_i\,(\hat{s}_i - \text{imp}_i)^2}{\sum_i m_i} $$

AdamW, lr=1e-3, 20 epoch, 80/20 train-val. F1'i de τ=0.5 ikili eşikte raporluyoruz.

### 3.6 Pruning kararı (çıkarım)
$$ \hat{C} = \{\,u_i \in C : \hat{s}_i \ge \tau \,\}, \qquad \tau = 0.5 $$
Hiçbir turn eşiği geçmezse → en yüksek skorlu turn tutulur (boş bağlam olmasın).

---

## 4. Değerlendirme & Baselines (≈45 sn)

Kısaltılmış bağlamı LLM'e verip cevap ürettiriyoruz, 4 yöntemi **adil** karşılaştırıyoruz
(hepsi CfC'nin tuttuğu **aynı k turn sayısını** tutar):

| Yöntem | Ne yapar |
|---|---|
| **Full** | Tüm bağlam (üst sınır kalite, alt sınır verimlilik) |
| **CfC** (bizim) | Öğrenilmiş önem skoruna göre tut |
| **Random** | Rastgele k turn (alt sınır baseline) |
| **Cosine** | Soruya SBERT-kosinüs benzerliği en yüksek k turn |

**Metrikler:** Kalite → ROUGE-L, BLEU-1/4, BERTScore-F1. Verimlilik → ortalama token sayısı,
**token azalması %**, TTFT (ilk token süresi).

> **Neden CfC > Cosine (anlatım noktası):** Cosine sadece soruya *yüzeysel benzerliği* ölçer.
> CfC ise öğretmenden **"hangi turn cevabı gerçekten değiştiriyor"** sinyalini öğrenir ve bunu
> içerik (embedding) + sürpriz (Δt) + sıralı bağlam (recurrent durum) ile birleştirir.

---

## 5. Sonuç anlatımı (rakamlar gelince doldur — ≈30 sn)
- "CfC, **~%XX token azalması** sağlarken ROUGE-L'i Full'e çok yakın tutuyor."
- "Aynı token bütçesinde CfC, **Random ve Cosine'dan yüksek** ROUGE-L veriyor → öğrenilmiş önem işe yarıyor."
- "Eğitim eğrisi: val loss düşüyor, F1 yükseliyor → CfC öğretmen sinyalini öğreniyor."
- (Varsa) "Ablation: τ arttıkça daha agresif pruning; τ≈0.5 kalite/verimlilik dengesi."

---

## 6. 4–5 dk için konuşma iskeleti
1. **Problem (20s):** uzun diyalog bağlamı pahalı; her turn eşit değil.
2. **Fikir (20s):** öğrenilmiş, sürekli-zamanlı pruning; sürprizi zamana gömüyoruz.
3. **Girdi (30s):** SBERT embedding + surprisal→Δt (denklem 3.1, 3.2).
4. **CfC (60s):** LTC ODE → kapalı-form; Δt kapısı (denklem 3.3). "Küçük, hızlı, sürekli zaman."
5. **Öğretmen (40s):** ΔNLL leave-one-out — "turn'ü at, cevap zorlaşıyor mu?" (denklem 3.4).
6. **Eğitim+Pruning (30s):** maskeli MSE, τ eşiği (3.5, 3.6).
7. **Sonuçlar (40s):** token azalması + ROUGE; Random/Cosine'ı geçiyoruz.
8. **Kapanış (10s):** "Bilgi-teorik zaman + öğrenilmiş önem = ucuz ama kaliteli bağlam."

---

## 7. Olası sorular (hazırlıklı ol)
- **Neden Δt'ye surprisal? Neden sabit zaman değil?** → Surprisal, turn'ün bilgi yoğunluğunun proxy'si;
  CfC'ye "bu turn önemli bir olay" sinyalini zaman üzerinden veriyoruz. Sabit zaman bu bilgiyi taşımaz.
- **CfC vs sıradan RNN/LSTM?** → CfC sürekli-zamanı *yerel* destekler (Δt girdi), çok küçük ve stabil;
  düzensiz zaman aralıklarına LSTM'den daha doğal.
- **Öğretmen neden 1.5B küçük model?** → Etiket sinyali *göreli* NLL farkı; küçük model bile "hangi turn
  cevabı değiştiriyor"u güvenilir sıralar. Hız için tercih ettik.
- **τ'yu nasıl seçtiniz?** → 0.5 varsayılan; ablation ile kalite/token dengesini doğruluyoruz.
- **Cosine zaten benzerlik buluyor, fark ne?** → Cosine soru-turn yüzey benzerliği; CfC *cevaba etki*yi
  öğrenir (bazen düşük-benzerlikli ama kritik bir turn'ü tutar, alakalı görünen gürültüyü atar).

---

### Anahtar denklemler (ezber kartı)
| | Denklem |
|---|---|
| Surprisal | `S(u) = -mean log P(w_t | w_<t)` |
| Δt | `Δt_i = 0.05 + 1.0 · rank(S_i)/N` |
| CfC | `x(t)=σ(-[w_τ+f]t)·g + (1-σ(...))·h` |
| Öğretmen | `imp_i = max(0, NLL(a|C\u_i) - NLL(a|C))` |
| Kayıp | `L = Σ m·(ŝ-imp)² / Σ m` |
| Pruning | `tut eğer ŝ_i ≥ τ (=0.5)` |
