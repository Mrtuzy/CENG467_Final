# CfC Kanıt-Suite'i — Rehber (entropi → zaman → CfC → pruning)

Bu rehber, **modeli eğitmeden önce** projenin merkez fikrinin — *kelimelerin/turn'lerin
entropisini (surprisal) zamana (Δt) çevirip CfC ile çözerek bağlam budama* — matematiksel
olarak sağlam mı, çürütülebilir mi olduğunu test eden simülasyon paketini ve bu
simülasyonların çıktılarıyla **modeli nasıl tasarlayacağımızı** anlatır.

İçindekiler:
1. [v1 neden yetersiz hissettiriyordu](#1-v1-neden-yetersiz)
2. [Yeni suite'in tasarım felsefesi](#2-tasarım-felsefesi)
3. [Matematiksel çekirdek](#3-matematiksel-çekirdek)
4. [Simülasyonlar tek tek](#4-simülasyonlar)
5. [Simülasyon → model tasarımı (asıl çıktı)](#5-simülasyon--model-tasarımı)
6. [Sunum anlatısı](#6-sunum-anlatısı)
7. [Çalıştırma ve sonraki adımlar](#7-çalıştırma-ve-sonraki-adımlar)

---

## 1. v1 neden yetersiz?

Eski `cfc_sims/` (artık `simulations/_archive_v1/`) dört varsayımı test ediyordu ve
hepsi "✓ tutarlı" çıkıyordu. Sorun tam da buydu — hem matematiksel hem sunumsal olarak:

| # | v1'in zayıflığı | Sonucu |
|---|-----------------|--------|
| A | **Substrat uyuşmazlığı.** Simülasyonlar LTC ODE kullanıyordu ama proje **CfC** eğitiyor. "Sims LTC, model CfC" açığı reviewer'ın ilk soracağı şeydi. | Mekanizmanın modelde geçerli olduğu *gösterilmemişti*. |
| B | **Tautoloji ölçmek.** Sim1'in "φ ↔ surprisal r=+0.89" bulgusu aslında tanım gereği: Δt=Δt_min+βS ve φ(Δt) monoton olduğundan korelasyon tasarımdan gelir. | Bir mekanizma *kararını* "ampirik bulgu" gibi sunmak güveni zayıflatır. |
| C | **Metrik seçiminde geriye-uydurma.** Sim1, ilk metrik (‖Δx‖) yanlış işaret verince φ'ye geçmişti. "İstediğimiz işareti alana kadar metriği değiştirdik" izlenimi. | Ön-kayıtlı (pre-registered), gerekçeli metrik yoktu. |
| D | **Planted-signal devresi.** Sim3/Sim4'te önem hem girdiye hem Δt'ye gömülüydü; sonuç kısmen tasarımdan geliyordu. Kanallar ayrıştırılmamıştı. | Avantajın *nereden* geldiği belirsizdi. |
| E | **İstatistik yok.** Tek seed (seed=0), error bar yok, güven aralığı yok, etki büyüklüğü yok. | Sayılar gürültü mü gerçek mi belirsizdi. |
| F | **"Her şey çalışıyor" anlatısı.** Hiç negatif/sınır sonucu yoktu. | Bilimsel olarak zayıf duruş; kırılma noktası gösterilmiyordu. |
| G | **Tasarıma bağlanmıyordu.** Simülasyonlar β, Δt_min, τ, eşleme fonksiyonu gibi **model kararlarını üretmiyordu** — sadece "doğrulama" yapıyordu. | Senin asıl hedefin ("çıktılarla modeli tasarla") karşılanmıyordu. |

---

## 2. Tasarım felsefesi

Yeni suite (`cfc_proof/`) yedi ilkeyle kuruldu:

1. **Türet, sonra ölç.** Tautolojiyi (B) ölçmek yerine *kapalı formda türet* ve asıl
   sorulması gerekeni ayır. φ(Δt)=1−e^{−λΔt} analitiktir; o yüzden "asıl kanıt" Δt
   mekanizması değil, surprisal'ın **önemle** korelasyonudur (→ S3).
2. **Substrat sadakati.** Mekanizmayı şeffaf analitik LTC'de türet, **gerçek ncps CfC**'de
   (eğitimsiz, sabit ağırlık) doğrula (A açığını kapatır → S0).
3. **Turn-başına okuma.** Gerçek `CfCPruner` her turn için skor verir; bu yüzden doğru
   büyüklük *son-duruma katkı* değil, turn anında okunabilir **bağlanma** `commit_i`'dir
   (`core.readout_signal`). Bu, D'deki kanal karışıklığını da çözer (içerik eşit →
   önem yalnızca Δt kanalından akar).
4. **Ön-kayıtlı, gerekçeli metrikler.** Her sim hipotezi + çürütme koşulunu *önce* yazar.
5. **Düzgün istatistik.** Çok seed, **bootstrap %95 GA**, Cohen's d, eşli testler (E).
6. **Dürüst sınır.** En az bir sim *negatif/kırılma* gösterir (S3'ün ρ* eşiği, S4'ün
   redundans-körlüğü) — bu, "her şey çalışıyor"dan bilimsel olarak güçlüdür (F).
7. **Tasarım çıktısı.** Her sim ya bir **model hiperparametresi** ya da bir **go/no-go
   koşulu** üretir (G). `run_all.py` bunları "MODEL TASARIM ÇIKTILARI" olarak basar.

---

## 3. Matematiksel çekirdek

**Sürekli-zaman bağlanma (commitment).** Sabit girdi I, Δt süresi boyunca tutulursa,
doğrusallaştırılmış sızdıran düğüm `dx/dt = −λ(x − x*(I))` *kapalı formda* çözülür:

```
x(Δt) = x* + (x0 − x*) e^{−λΔt}
⇒  φ(Δt) ≜ ‖x(Δt) − x0‖ / ‖x* − x0‖ = 1 − e^{−λΔt}.
```

φ, **monoton-artan ve içbükey**: Δt→0'da 0, Δt→∞'da 1'e doyar. Δt = Δt_min + β·S
eşlemesi altında φ, surprisal S'in monoton fonksiyonudur. Bu kapalı form, S0'da gerçek
ncps CfC'nin bağlanma eğrisiyle **ρ=+0.94** örtüşür → analitik substrat modele sadıktır.

**Δt kaldıracının duyarlılığı** (S1): `dφ/dS = λβ·e^{−λΔt} = λβ(1−φ)`. Yani kaldıraç
yüksek surprisal'da (φ→1) **tükenir** — doyum. Bu, neden ham doğrusal eşlemenin aykırı
surprisal'da bozulduğunu (S5) önceden haber verir.

**Turn-başına okunabilir sinyal** (`core.readout_signal`): `s_i = commit_i = 1−e^{−λΔt_i}`
(+ gürültü tabanı σ). Önem-kurtarma = `Spearman(s_i, u_i)`. Uniform-Δt ⇒ commit sabit ⇒
kurtarma ≈ 0. Entropi-Δt ⇒ commit, S üzerinden önemi taşır ⇒ kurtarma ≈ ρ(S,u).

**Sabit-bütçe (S5):** ΣΔt sabit tutulursa eşlemenin *ölçeği* değil *şekli* ölçülür;
aykırı bir surprisal doğrusal eşlemede bütçeyi kapıp diğer önemli turn'leri aç bırakır.

**Teacher hedefi (S4):** Cevap, fact-başına **doyan** kanıtla modellenir
`p = softmax(θ·tanh(evidence_f/θ))`. Leave-one-out KL, bir turn'ün **marjinal cevap
etkisini** ölçer; total-variation (TV) ile ρ≈1.0 örtüşür (redundanstan bağımsız).

---

## 4. Simülasyonlar

Her sim: hipotez → çürütme koşulu → ölçüm (çok seed + bootstrap GA) → figür → karar.
Tümü modelsiz; yalnızca `numpy`, `scipy`, `matplotlib` (S0 ek olarak `torch`+`ncps`).

| Sim | Test eder | Anahtar sonuç | Üretilen tasarım kararı |
|-----|-----------|---------------|--------------------------|
| **S0** `s0_substrate_faithfulness` | Analitik LTC ↔ gerçek CfC sadakati | φ kapalı-formu birebir; CfC ile ρ=+0.94 | Analitik substrat = sadık vekil |
| **S1** `s1_mechanism_analytic` | Entropi→Δt mekanizması (tautoloji değil) | φ analitik; %32 turn doyumda | Asıl kanıt S3'e devredilir; doyum riski → S5 |
| **S2** `s2_beta_dtmin_design` | β, Δt_min'in anlamlı orta rejimi | β*=1.0, Δt_min*=0.05; β=0'a göre +0.56 | **β, Δt_min önerisi** |
| **S3** `s3_proxy_quality_limit` | surprisal ↔ ÖNEM kritik eşiği | ρ*≈0.1 (>uniform); ρ≥0.5'te >cosine | **go/no-go**: QReCC'de ρ ölçülmeli |
| **S4** `s4_loo_kl_redundancy` | Leave-one-out KL neyi ölçer | TV ile ρ≈1.0; a-kurtarma redundansla 0.81→0.54 | Hedef = marjinal gereklilik (doğru) |
| **S5** `s5_mapping_function_design` | Δt eşleme fonksiyonu | rank en dayanıklı (aykırıda) | **eşleme: 'rank'** |
| **S6** `s6_endtoend_pruning` | Sadakat–sıkıştırma (uçtan uca) | %90 sadakatte %40 sıkıştırma (uniform %9) | **τ≈0.48**; başlık sonucu |

### S0 — Substrat sadakati
Mekanizmanın LTC'ye özgü bir yapay sonuç olmadığını kanıtlar. Analitik φ=1−e^{−λΔt}
hem sayısal çözümle birebir (sapma ~1e-10) hem de 12 rastgele ağırlıklı **gerçek ncps
CfC** örneğinin ortalama bağlanma eğrisiyle (ρ=+0.94) örtüşür. → Kalan sim'lerde şeffaf
analitik substratı güvenle kullanırız. **Bu, v1'in en büyük açığını (LTC≠CfC) kapatır.**

### S1 — Mekanizma analitik
φ(Δt) ve dφ/dS=λβ(1−φ) türetilir. Mesaj: "φ↔surprisal korelasyonu" bir *bulgu* değil
*tasarım sonucudur*; ve yüksek surprisal'da kaldıraç doyar (turn'lerin %32'si φ≥0.95).
Asıl kanıt yükü S3'e taşınır. Bu, v1'in B ve C zayıflıklarını dürüstçe düzeltir.

### S2 — β ve Δt_min'i simülasyon seçer
(β, Δt_min) ızgarasında önem-kurtarmayı (ρ=0.6 sabit vekil, içerik eşit, gürültü tabanlı)
maksimize eden bölge aranır. **β*=1.0 config varsayılanını doğrular**; Δt_min*=0.05
varsayılan 0.1'i düşürmeyi önerir. β=0 (entropi yok) kurtarmayı çökertir (+0.006) — kaldıraç
gerçek. Bu aynı zamanda **ablation (§7)** için doğal gerekçedir.

### S3 — Asıl bilimsel sonuç: proxy kalitesi sınırı
Tüm yöntemin tek kritik riski: *surprisal, önemin iyi bir vekili değilse hiçbir şey
kazanılmaz.* ρ(S,u) süpürülür. Entropi-Δt kurtarması ρ ile **0→0.97** düzgün yükselir;
uniform ~0'da sabit; cosine baseline'ı ρ≥0.5'te geçer. **Kritik ρ*≈0.1** entropi'nin
uniform'u geçtiği eşik. → Modeli eğitmeden önce QReCC'de `corr(surprisal, teacher-önem)`
ölçülmeli (go/no-go). Bu, v1'de hiç olmayan **dürüst kırılma noktasıdır**.

### S4 — Leave-one-out KL'in gerçekte ölçtüğü şey
Fact-başına doyan cevap modelinde redundans devreye girer: redundant bir turn'ü çıkarmak
cevabı az değiştirir (bilgi başka turn'de hayatta). Sonuç: KL, **marjinal cevap-gerekliliğini**
(TV ile ρ≈1.0) redundanstan bağımsız sadakatle ölçer; ham latent önem `a` kurtarması ise
redundansla düşer (0.81→0.54). **Bu bir hata değil:** pruning için doğru hedef tam da
marjinal gerekliliktir — redundant turn'ü budamak güvenlidir. [0,1] min-max normalizasyon
sırayı korur (ρ=+1.0). v1'in tautolojik Sim4'ünü nüanslı bir sonuca çevirir.

### S5 — Δt eşleme fonksiyonunu simülasyon seçer
Sabit hesap bütçesinde linear/bounded/rank eşlemeleri, normal ve **aykırı** surprisal
altında karşılaştırılır. Ham DistilGPT-2 surprisal'ı ağır kuyrukludur; doğrusal eşleme
aykırıda en çok kaybeder. **'rank' (sıra-normalize) en dayanıklı** → `build_inputs.py`'de
surprisal→Δt adımının rank-temelli yapılması önerilir.

### S6 — Uçtan uca pruning (sunum başlığı)
İzleyicinin umursadığı düzey: *ne kadar budarsak cevabı koruyan turn'leri ne kadar tutarız?*
Sadakat (gerekli turn recall) – sıkıştırma eğrisi. **Entropi-CfC vekili %90 sadakatte %40
sıkıştırmaya** izin verir; uniform %9, cosine %31, random %10. **Önerilen τ≈0.48** (config
varsayılanı 0.5'i doğrular). Bu eğri raporun/sunumun **merkez görselidir**.

---

## 5. Simülasyon → model tasarımı

Senin asıl hedefin buydu: *önce simülasyonla kanıtla, sonra çıktılarıyla modeli tasarla.*
İşte somut köprü — her satır `results.json`'dan okunabilir ve doğrudan `src/config.py` /
`src/build_inputs.py`'ye uygulanır:

| Karar | Sim | Öneri | Mevcut config | Aksiyon |
|-------|-----|-------|---------------|---------|
| `BETA` | S2 | **1.0** | 1.0 | Değişiklik yok — doğrulandı ✓ |
| `DELTA_T_MIN` | S2 | **0.05** | 0.1 | 0.05'e düşürmeyi dene (ablation'a ekle) |
| surprisal→Δt eşlemesi | S5 | **rank** | linear | `build_inputs.py`'de rank-normalize ekle |
| `TAU` (pruning eşiği) | S6 | **≈0.48** | 0.5 | 0.5 yeterince yakın — koru, 0.45–0.50 sweep |
| Go/No-Go (eğitim öncesi) | S3 | **ρ(S, teacher-önem) > 0.1** | — | QReCC'de ÖLÇ; ρ büyükse entropi-CfC, küçükse cosine baseline |
| Teacher hedefi yorumu | S4 | marjinal gereklilik | leave-one-out KL ✓ | Yorumu rapora yaz; redundant→düşük hedef beklenir |
| Substrat gerekçesi | S0 | CfC ≈ analitik LTC | CfC (ncps) | §6'da mimari gerekçe olarak kullan |
| Mimari kanal | S3/S4 | Δt kanalı önemi taşıyor; cosine ucuz alternatif | — | **Baseline olarak cosine'i tut** (S3 sınırı için sigorta) |

**Önerilen tasarım iş akışı:**
1. `build_inputs.py`'de surprisal'ı hesapla, **`corr(surprisal, teacher_targets)`'ı ölç**
   (S3 go/no-go). ρ > ~0.1–0.3 ise devam; değilse fikri gözden geçir / cosine'e dön.
2. surprisal→Δt'yi **rank-normalize** yap (S5).
3. `config.py`: BETA=1.0 (sabit), DELTA_T_MIN ∈ {0.05, 0.1} ablation, TAU ∈ {0.45,0.5}.
4. CfC'yi eğit (`train_cfc.py`); §7 ablation'ı S2/S5/S6 sweep'leriyle hizala.
5. Değerlendirmede S6 eğrisinin gerçek QReCC karşılığını (sadakat–sıkıştırma) çiz.

---

## 6. Sunum anlatısı

Mantıksal akış (slayt sırası önerisi):

1. **Problem & fikir** — uzun bağlam pahalı; insan bilişi sürpriz bilgiye daha çok
   "işlem zamanı" ayırır → surprisal'ı Δt'ye çevirip CfC ile çöz.
2. **S0** — "Mekanizma gerçek mi, modele uyuyor mu?" → analitik φ kapalı-formu = gerçek
   CfC (ρ=0.94). Substrat sadık. *(LTC≠CfC itirazını baştan kapatır.)*
3. **S1** — "Entropi→Δt neden işler?" → φ=1−e^{−λΔt} türet; ama dürüstçe: bu *mekanizma*,
   *kanıt değil*. Asıl soru: surprisal önemi taşıyor mu?
4. **S3** — **anlatının kalbi**: kazanç, surprisal↔önem korelasyonuna bağlı; kritik ρ*
   var. "Her şey çalışıyor" demiyoruz; *hangi koşulda* çalıştığını söylüyoruz.
5. **S4** — teacher etiketimiz (leave-one-out KL) "ham önem"i değil "marjinal cevap
   gerekliliği"ni ölçer — pruning için doğru hedef; redundant turn'ü budamak güvenli.
6. **S2 + S5** — hiperparametreleri (β, Δt_min, eşleme) *veriyle* seçtik; keyfi değil.
7. **S6** — **başlık görseli**: %90 sadakatte %40 sıkıştırma, baseline'ları geçiyor.
8. **Kapanış** — "Bu simülasyonlar modeli eğitmeden mekanizmayı kanıtladı ve config'i
   tasarladı; sıradaki adım QReCC'de ρ go/no-go testi ve eğitim."

Vurgulanacak güç: **dürüstlük** (S3 ρ*, S4 redundans, S1 doyum) — kırılma noktalarını
gösteren bir suite, "hepsi yeşil" bir suite'ten daha inandırıcıdır.

---

## 7. Çalıştırma ve sonraki adımlar

```bash
conda activate jepa_conda           # numpy, scipy, matplotlib, torch, ncps
cd simulations/cfc_proof
python run_all.py                   # 7 sim + figürler + results.json + tasarım çıktıları
python s3_proxy_quality_limit.py    # tek tek de çalışır
```

- Figürler: `figures/s0…s6_*.png` (raporda §6/§7 için).
- Sayısal sonuçlar: `results.json`.
- Bulgu özeti: [`RESULTS.md`](RESULTS.md).

**Sonraki adım (modeli tasarlamak):** §5 tablosunu uygula. En kritik tek iş, **S3
go/no-go testi**: `build_inputs.py` çıktısı üzerinde `scipy.stats.spearmanr(surprisal,
teacher_targets)` hesapla. Bu sayı, tüm yöntemin eğitime değer olup olmadığını eğitimden
önce söyler.
