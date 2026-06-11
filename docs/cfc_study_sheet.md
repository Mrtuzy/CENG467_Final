# CfC Context Pruning — Çalışma & Sunum Sheet'i
*(4–5 dk'lık sunum için hazırlanmış; CfC kısmı senin)*

---

## 0. Tek cümlelik özet
> Bir konuşmadaki **hangi geçmiş turn'lerin gerçekten gerekli** olduğunu öğrenen, sürekli-zamanlı
> küçük bir sinir ağı (CfC) eğitiyoruz; gereksiz turn'leri atıp (pruning) LLM'e daha kısa bağlam
> vermeyi hedefledik; fakat gerçek QReCC deneyi gösterdi ki **DistilGPT-2 surprisal'ı turn öneminin
> güvenilir proxy'si değil**, bu yüzden mevcut CfC eğitimi başarısız oldu.

---

## 1. Problem & Motivasyon (≈30 sn)

- Çok-turlu diyalogda (QReCC veri seti) LLM'e **tüm geçmişi** vermek pahalı: uzun bağlam = çok token
  = yüksek gecikme (TTFT) ve maliyet.
- Ama her turn eşit değerli değil — bazıları cevabı belirler, çoğu gürültü.
- **Amaç:** "context pruning" — cevabı bozmadan bağlamı kısalt.
- **Bizim katkımız:** Bunu **sürekli-zamanlı, hafif bir ağ (CfC)** ile yapan, **öğrenilmiş bir pruner**.
  Girdisi turn'lerin **anlamsal temsili (SBERT)**; ayrıca bir **bilgi-teorik sinyali (surprisal → zaman)**
  hipotez olarak test ediyoruz.

> ⚠️ **Dürüst bulgu (sunumun ana noktası):** "Surprisal, turn önemini öngörür mü?" hipotezini
> eğitimden önce ölçtük. Gerçek QReCC teacher label'larında ortalama Spearman ρ = **-0.074**.
> Bu, simülasyonlarda belirlenen **go/no-go eşiği ρ\*=0.1'in altında**. Yani entropy→Δt fikrinin
> kritik varsayımı kırıldı; eğitim metrikleri de bunu doğruladı.

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
3. **Öğretmen etiketi:** bir LLM ile her turn'ün **cevap-koşullu önemi** `imp_i` (aşağıda).
4. **CfC eğitimi:** `(e_i, Δt_i) → ŝ_i` öğren, hedef `imp_i` (regresyon).
5. **Pruning + Değerlendirme:** `ŝ_i ≥ τ` olan turn'leri tut, LLM'e ver, kaliteyi/verimliliği ölç.

> Bu koşuda 5. aşama tamamlanmadı: `eval_results.json` oluşmadı. Elimizde güvenle raporlanabilecek
> çıktı: proxy go/no-go figürü, train/val loss, validation F1/precision/recall ve validation MAE.

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

**Bu koşudaki eğitim özeti:**
- Train loss: **0.1293 → 0.1201** düştü.
- Val loss: en iyi **0.1251** (epoch 6), final **0.1257**; yani genelleme iyileşmedi.
- Val MAE: yaklaşık **0.300** civarında yatay kaldı.
- En iyi val F1: **0.050** (epoch 19); precision **0.397**, recall sadece **0.027**.

Yorum: model az sayıda turn'e "önemli" diyor, bu yüzden precision fena görünmüyor; fakat asıl
önemli turn'lerin neredeyse tamamını kaçırıyor. Pruning için ölümcül olan kısım düşük recall.

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

## 5. Gerçek Sonuç Anlatımı (≈45 sn)
- "Simülasyonlar mekanizmanın çalışabileceğini gösterdi, ama bunun tek kritik şartı vardı:
  surprisal'ın gerçek önemle pozitif korele olması."
- "QReCC üzerinde go/no-go testi bu şartın sağlanmadığını gösterdi: ortalama Spearman
  **ρ=-0.074**, gerekli eşik **ρ\*=0.1**."
- "Buna rağmen CfC'yi eğittik; eğitim loss'u düştü ama validation loss iyileşmedi, MAE
  **0.300** civarında kaldı."
- "τ=0.5 ile en iyi F1 sadece **0.050**; recall **0.027**. Yani model önemli turn'leri tutamıyor."
- "Bu yüzden CfC için final ROUGE/token reduction iddiası yapmıyoruz. Bu koşu pozitif sonuç değil,
  hipotezi test eden ve başarısızlık nedenini gösteren negatif sonuç."

### Neden başarısız olduk?
1. **Proxy varsayımı yanlış çıktı:** DistilGPT-2 surprisal'ı QReCC'deki cevap-koşullu turn önemini
   taşımıyor. Şaşırtıcı bir turn cevap için gerekli olmayabiliyor; gerekli bir turn de sıradan
   dilsel biçimde yazılmış olabiliyor.
2. **Öğretmen hedefi seyrek ve zor:** Leave-one-out ΔNLL çoğu turn'e düşük skor veriyor; pozitif
   turn sayısı az olunca τ=0.5 classifier recall'ı çöküyor.
3. **CfC sinyal yerine prior öğreniyor olabilir:** Train loss azalıyor ama val loss sabit/kötüleşiyor;
   bu, modelin genellenebilir önem sinyali yakalamadığını düşündürüyor.
4. **Downstream değerlendirme yok:** `eval_results.json` oluşmadığı için "CfC şu kadar token azalttı"
   veya "ROUGE-L'i korudu" demek doğru değil.

---

## 6. 4–5 dk için konuşma iskeleti
1. **Problem (20s):** uzun diyalog bağlamı pahalı; her turn eşit değil.
2. **Fikir (20s):** öğrenilmiş, sürekli-zamanlı pruning; sürprizi zamana gömüyoruz.
3. **Girdi (30s):** SBERT embedding + surprisal→Δt (denklem 3.1, 3.2).
4. **CfC (60s):** LTC ODE → kapalı-form; Δt kapısı (denklem 3.3). "Küçük, hızlı, sürekli zaman."
5. **Öğretmen (40s):** ΔNLL leave-one-out — "turn'ü at, cevap zorlaşıyor mu?" (denklem 3.4).
6. **Go/no-go (40s):** gerçek QReCC'de ρ=-0.074; kritik varsayım kırıldı.
7. **Eğitim sonucu (40s):** train loss düşüyor ama val loss/MAE/F1 başarısız; recall çok düşük.
8. **Kapanış (10s):** "Mekanizma değil, proxy seçimi kırıldı; sonraki adım soru-koşullu/öğretmen-koşullu proxy."

---

## 7. Olası sorular (hazırlıklı ol)
- **Neden Δt'ye surprisal? Neden sabit zaman değil?** → Hipotezimiz buydu: surprisal bilgi yoğunluğu
  proxy'si olabilir ve CfC'ye "bu turn önemli bir olay" sinyali verebilir. Ama gerçek QReCC ölçümü
  bunun çalışmadığını gösterdi; bu yüzden artık surprisal'ı tek başına savunmuyoruz.
- **CfC vs sıradan RNN/LSTM?** → CfC sürekli-zamanı *yerel* destekler (Δt girdi), çok küçük ve stabil;
  düzensiz zaman aralıklarına LSTM'den daha doğal.
- **Öğretmen neden 1.5B küçük model?** → Etiket sinyali *göreli* NLL farkı; küçük model bile "hangi turn
  cevabı değiştiriyor"u güvenilir sıralar. Hız için tercih ettik.
- **τ'yu nasıl seçtiniz?** → 0.5 varsayılan eşik; bu koşuda kötü çalıştı. En iyi F1 0.050 ve recall
  0.027 olduğu için sonraki denemede top-k/ranking loss veya validation-tuned threshold gerekir.
- **Cosine zaten benzerlik buluyor, fark ne?** → İdeal olarak CfC cevaba etkiyi öğrenir; ama bu koşuda
  CfC bunu öğrenemedi. Cosine baseline'ı geçme iddiası yok çünkü downstream eval tamamlanmadı.
- **Bu tamamen başarısız mı?** → Son ürün olarak evet, başarılı pruner değil. Bilimsel süreç olarak
  faydalı: simülasyondaki go/no-go kriteri gerçek veride hipotezi reddetti ve eğitim eğrileri bunu
  doğruladı.

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
