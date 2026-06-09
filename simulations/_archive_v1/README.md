# Kavram-Doğrulama Simülasyonları

Bu klasördeki simülasyonlar **model eğitmez ve LLM yüklemez**. Amaçları,
projenin temel tasarım varsayımlarının *kavramsal olarak sağlam mı, çürütülebilir
mi* olduğunu saf matematikle (yalnızca `numpy` + `matplotlib`) test etmektir.

Her simülasyon: bir hipotez tanımlar → çürütme koşulunu belirtir → ölçer →
bir figür kaydeder (`figures/`) → **TUTARLI / TUTARSIZ** kararı verir.

## Çalıştırma

```bash
python simulations/run_all.py        # hepsi + özet + results.json
python simulations/sim1_entropy_dt_mapping.py   # tek tek de çalışır
```

Bağımlılık: `numpy`, `matplotlib`. (GPU/model gerekmez.)

## Simülasyonlar

| # | Test edilen varsayım | Çürütme koşulu | Sonuç |
|---|----------------------|----------------|-------|
| **Sim1** | Δt = Δt_min + β·S eşlemesi yüksek-surprisal turn'lere daha güçlü "bağlanma" (denge-noktasına yaklaşma φ) sağlar | φ ↔ surprisal korelasyonu ≈ 0, ya da uniform-Δt ile aynı | ✓ r=+0.89 vs uniform +0.20 |
| **Sim2** | β, entropi-hassasiyetini monoton kontrol eder ve dejenere değildir | ayrım β'dan bağımsız / monoton azalan | ✓ β=0'da ~0, β↑ ile artıp doyuyor |
| **Sim3** | Sürekli-zaman (entropi-Δt), önem sinyalini uniform-adıma göre daha sadık korur | Spearman kazancı ≤ 0 | ✓ ort. +0.24 kazanç, gürültüyle azalıyor |
| **Sim4** | Leave-one-out KL, gerçek (latent) önem sıralamasını kurtarır | KL ↔ importance Spearman ≈ 0 | ✓ ort. +0.91, %100 pozitif |

## Yöntem notları

- **LTC ODE** (`sim_common.LTCCell`): `dx/dt = -x/τ + S(x,I)⊙(A-x)`, sabit
  (öğrenilmemiş) ağırlıklarla. Amaç dinamiğin *niteliksel* davranışı, performans
  değil. Her turn kendi Δt'si boyunca entegre edilir.
- **Bağlanma oranı** `φ = ‖x_sonra − x_önce‖ / ‖x* − x_önce‖ ∈ [0,1]`: Δt etkisini
  denge-noktası uzaklığından ayırmak için kullanılır (ham ‖Δx‖ ikisini karıştırır;
  bkz. Sim1 docstring). Δt → ∞ iken φ → 1.
- **Sim4** cevabı `p = softmax(Σ a_i·v_i)` ile modeller; leave-one-out KL'i latent
  önem `a_j`'ye karşı ölçer — LLM olmadan teacher-etiketleme önermesini sınar.

## Rapora katkı (LNCS)

Bu simülasyonlar §6/§7'de "tasarım gerekçelendirmesi" olarak kullanılabilir:
ana deneyler *ne kadar* iyi çalıştığını gösterirken, bu simülasyonlar *neden*
çalışmasını beklediğimizi — mekanizmanın kendisinin sağlam olduğunu —
gerçek modelden bağımsız, kontrollü ve çürütülebilir biçimde kanıtlar.
Sim3 ve Sim4'teki gürültü-degradasyon eğrileri, varsayımların hangi koşulda
zayıfladığını da dürüstçe gösterir.
