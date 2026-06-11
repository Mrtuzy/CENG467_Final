# CfC Bölümü — Sunum Konuşma Metni
*(~4.5 dk, 13 slayt. Köşeli parantezler = sahne yönergesi, okunmaz.)*

---

### 1. CfC Extension: The Core Idea  *(~25 sn)*
"Şimdi projenin keşifsel kısmına geliyorum: entropi-güdümlü, sürekli-zamanlı bir pruner.
Fikrin çıkış noktası bilişsel: insanlar yoğun, bilgi dolu metin okurken **yavaşlar** — sürpriz,
beklenmedik kelimeler daha fazla işlem süresi ister. Biz de sorduk: aynı sinyali kullanıp bir
konuşmadaki **hangi turn'ün daha fazla 'işlem zamanı' hak ettiğine** karar verebilir miyiz?
Sabit eşikler bunu bilemez; biz bunu öğrenmeye çalışıyoruz."

### 2. Cognitive Motivation: Reading Time ∝ Surprisal  *(~20 sn)*
"Bunun psikodilbilimde sağlam bir temeli var: bir kelimenin okuma süresi, surprisal'ıyla —
yani eksi-log-olasılığıyla — orantılı. Levy 2008 ve Smith-Levy 2013 bunu gösteriyor.
NLP'ye taşıdığımız hipotez şu: bir turn sözlüksel olarak şaşırtıcıysa muhtemelen **yeni bilgi**
getiriyor, dolayısıyla cevaba daha önemli olabilir. Dikkat: surprisal *eşittir* önem demiyoruz;
**işe yarar bir vekil mi** diye soruyoruz — ve eğitmeden önce test ediyoruz."

### 3. From Reading Time to the CfC Time Step  *(~25 sn)*
"Mekanik olarak: her turn için DistilGPT-2 ile surprisal hesaplıyoruz. Sonra bunu, CfC'nin
'geçen zaman' adımına çeviriyoruz — aykırı değerlere dayanıklı olsun diye **sıra-normalize**
ederek: Δt = Δt_min + β·rank(S)/T. CfC bu Δt üzerinden integre ediyor. Büyük Δt → durum
daha güçlü 'kilitleniyor' → daha yüksek önem skoru. Yani sürpriz, doğrudan ağın hafıza
dinamiğine giriyor."

### 4. Why Simulations Before Training?  *(~20 sn)*
"Burada ciddi bir risk var: surprisal, QReCC'te gerçek turn önemiyle **hiç** korele olmayabilir.
Eğer korelasyon sıfırsa, entropi hiçbir sinyal katmaz ve 7B öğretmenle eğitim hesabını boşa
harcarız. O yüzden eğitmeden önce **yedi model-bağımsız simülasyon** yaptık; her biri tek bir
tasarım sorusunu izole edip ya somut bir değer ya da yanlışlanabilir bir go/no-go eşiği döndürüyor."

### 5. Simulation Suite: a Roadmap  *(~15 sn — hızlı geç)*
"İşte yedi simülasyonun özeti — geçerlilik, tasarım, etiket ve kritik test sorularını kapsıyor.
Zaman için ben sadece **projeyi belirleyen ikisine** odaklanacağım: S2 ve S3."

### 6. S2 — Choosing β and Δt_min  *(~25 sn)*
"S2 iki şey yapıyor. Solda, β ve Δt_min ızgarasında en iyi geri-kazanım β=1.0, Δt_min=0.05'te.
Ama asıl kritik nokta **sağda**: β=0 olduğunda — yani Δt sabit, entropi yok — geri-kazanım
neredeyse **sıfır**. Bu çok önemli: sinyali taşıyan şey **entropi terimi**, ağın kendisi değil.
Yani tüm yöntem, surprisal'ın iyi bir sinyal olmasına bağlı."

### 7. S3 — The Honest Go/No-Go  *(~25 sn)*
"S3 de tam bu varsayımı test ediyor. Entropi-Δt'nin başarısı, surprisal ile gerçek önem
arasındaki korelasyon ρ'ya bağlı. İki eşik çıkıyor: ρ yaklaşık 0.1'i geçmezse uniform'dan bile
iyi değil; cosine baseline'ı geçmek için ρ ≥ 0.5 lazım. Karar kuralı net: **gerçek QReCC'te ρ'yu
ölç, eşiğin altındaysa eğitimi iptal et.** Dürüst, yanlışlanabilir bir test."

### 8. Real QReCC Go/No-Go Result  *(~30 sn — vurgu noktası)*
"Ve gerçek veride varsayım **çöktü**. Üretilen QReCC öğretmen etiketlerinde ortalama Spearman
korelasyon **eksi 0.074**. Yani sadece eşiğin altında değil — **hafif negatif**. Cosine'ı geçme
hedefi olan 0.5'ten ise çok uzak. Surprisal ile cevaba-etki, bu veri setinde birbirinden bağımsız
çıktı. Buna rağmen, bir stres testi olarak CfC'yi yine de eğittik."

### 9. Architecture  *(~20 sn)*
"Mimari kısaca: her turn için SBERT embedding artı sıra-normalize Δt giriyor; CfC hücresi durumu
Δt üzerinden integre ediyor; turn başına sigmoid ile [0,1] önem skoru çıkıyor. Öğretmen etiketi,
gold cevabın **cevap-koşullu leave-one-out ΔNLL**'si — hafif bir LLM, Qwen-1.5B ile. Kayıp MSE,
eşik τ=0.5."

### 10. CfC Training Curves  *(~25 sn)*
"Eğitim eğrileri hikayeyi anlatıyor. Train loss 0.129'dan 0.120'ye düşüyor — yani model bir şeyler
ezberliyor. Ama **validation loss neredeyse hiç iyileşmiyor**; en iyi değer epoch 6'da 0.1251,
sonra yukarı sürükleniyor. Bu klasik bir işaret: model genellenebilir bir önem sinyali öğrenmiyor,
hedefin ortalamasına geriliyor."

### 11. CfC Validation Metrics  *(~25 sn)*
"Metrikler bunu doğruluyor. τ=0.5'te en iyi F1 sadece **0.05**; recall **0.027**. Yani model
gerçekten önemli turn'lerin neredeyse tamamını kaçırıyor. Precision 0.40'a çıkıyor ama bu sadece
modelin çok az turn'e 'önemli' demesinden. MAE 0.30 civarında düz — kalibre per-turn skor yok.
Pruning için ölümcül olan kısım bu düşük recall."

### 12. Why the CfC Run Failed  *(~25 sn — analiz)*
"Özetle başarısızlığın dört noktası: bir, zayıf vekil kanal — surprisal-önem korelasyonu eksi 0.074,
go/no-go eşiğinin altında. İki, eşik uyuşmazlığı — τ=0.5'te recall 0.03'ün altında. Üç, faydalı
validation kazancı olmadan ezberleme. Dört, downstream karşılaştırma tamamlanmadı; ROUGE ya da
token-azaltma iddiası yapamıyoruz. **Önemli ayrım:** CfC mekanizması matematiksel olarak hâlâ
geçerli — bu deney sadece, DistilGPT-2 surprisal'ının QReCC için güvenilir bir önem vekili olduğu
**daha güçlü iddiasını reddediyor**."

### 13. CfC Evaluation Plan  *(~20 sn — kapanış)*
"Mevcut durum dürüstçe bir negatif sonuç: kod ve notebook proxy-kontrolü ve eğitimi uçtan uca
çalıştırıyor, ama final downstream değerlendirme henüz yok. Sonraki adım açık: surprisal'ı daha
güçlü, **soruya-koşullu** bir önem vekiliyle değiştirmek ve ancak go/no-go testini geçince
baseline'lara karşı koşmak. Teşekkürler."

---

## Tek cümlelik kapanış (ezberle)
> "Mekanizma sağlam, ama bu deney bize şunu öğretti: öğrenilmiş pruning, ham surprisal'dan değil,
> **soruya-koşullu** bir sinyalden beslenmeli — go/no-go testimiz bunu eğitimden önce yakaladı."

## Zamanlama özeti
| Bölüm | Slayt | Süre |
|---|---|---|
| Fikir + motivasyon | 1–3 | ~70 sn |
| Simülasyon mantığı | 4–5 | ~35 sn |
| Kritik 2 sim (S2, S3) | 6–7 | ~50 sn |
| Gerçek sonuç (go/no-go) | 8 | ~30 sn |
| Mimari + eğitim + metrik | 9–11 | ~70 sn |
| Neden başarısız + plan | 12–13 | ~45 sn |
| **Toplam** | | **~5 dk** |

> 4 dk'ya sığdırman gerekirse: slayt 5'i tek cümleye indir, slayt 2 ve 3'ü birleştir, slayt 13'ü
> sadece son cümleyle geç.
