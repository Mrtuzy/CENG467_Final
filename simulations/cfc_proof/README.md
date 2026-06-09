# CfC Kanıt-Suite'i (`cfc_proof/`)

Modeli **eğitmeden önce**, projenin merkez fikrinin — *entropi (surprisal) → Δt → CfC →
pruning* — matematiksel olarak sağlam mı olduğunu test eden ve **modelin tasarım
parametrelerini üreten** modelsiz simülasyonlar. (v1 `simulations/_archive_v1/`'e taşındı.)

- 📘 **Rehber + tasarım köprüsü:** [`GUIDE.md`](GUIDE.md) — v1 neden yetersizdi, yeni
  suite'in matematiği, simülasyon→config haritası, sunum anlatısı.
- 📊 **Bulgular:** [`RESULTS.md`](RESULTS.md)
- 🔢 **Ham sonuçlar:** `results.json` · **Figürler:** `figures/`

## Çalıştırma
```bash
conda activate jepa_conda      # numpy, scipy, matplotlib (+ S0 için torch, ncps)
cd simulations/cfc_proof
python run_all.py              # 7 sim + figürler + results.json + tasarım çıktıları
python s3_proxy_quality_limit.py   # her sim tek tek de çalışır
```

## Simülasyonlar
| | Dosya | Test eder | Çıktı |
|--|-------|-----------|-------|
| S0 | `s0_substrate_faithfulness.py` | analitik LTC ↔ gerçek ncps CfC sadakati | substrat doğrulama |
| S1 | `s1_mechanism_analytic.py` | entropi→Δt mekanizması (kapalı form, tautoloji değil) | doyum analizi |
| S2 | `s2_beta_dtmin_design.py` | β, Δt_min'in anlamlı rejimi | **β\*, Δt_min\*** |
| S3 | `s3_proxy_quality_limit.py` | surprisal↔önem kritik korelasyonu | **go/no-go ρ\*** |
| S4 | `s4_loo_kl_redundancy.py` | leave-one-out KL neyi ölçer (redundans) | hedef yorumu |
| S5 | `s5_mapping_function_design.py` | Δt eşleme fonksiyonu (linear/bounded/rank) | **eşleme önerisi** |
| S6 | `s6_endtoend_pruning.py` | sadakat–sıkıştırma (uçtan uca vekil) | **τ\*** + başlık figürü |

`core.py`: ortak substrat (analitik LTC + gerçek CfC + kapalı-form etki/okuma),
Δt eşlemeleri, istatistik (bootstrap GA, Cohen's d) ve figür yardımcıları.

## Bir cümlede sonuç
Mekanizma modele sadık (S0) ve analitik (S1); β=1.0/Δt_min≈0.05 en iyi (S2); avantaj
surprisal↔önem korelasyonuna bağlı, **kritik ρ\* var** (S3); teacher hedefi marjinal
gerekliliği ölçer (S4); 'rank' eşlemesi dayanıklı (S5); **%90 sadakatte %40 sıkıştırma**,
τ≈0.48 (S6). Ayrıntı ve modeli buradan tasarlama adımları için → [`GUIDE.md`](GUIDE.md).
