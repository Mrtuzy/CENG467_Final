"""
Simülasyon 3 — Sürekli-zaman, önem sinyalini uniform-adıma göre daha iyi
koruyor mu? (Liquid'in asıl iddiası)
========================================================================
HİPOTEZ
    Bir diyalogda turn'lerin GERÇEK önemi (planted importance) düzensiz dağılır
    ve surprisal bunun gürültülü bir proxy'sidir. Entropi-Δt'li LTC, her turn'ün
    durum izini (‖Δx‖) gerçek önemle, uniform-Δt'li (discrete-RNN benzeri)
    versiyondan DAHA YÜKSEK sıra-korelasyonuyla eşler.

NEDEN ÖNEMLİ
    Projenin "neden liquid / neden sürekli-zaman?" iddiası tam da budur:
    turn'ler arası zamanı entropiyle modüle etmek, önemli turn'leri dolgudan
    ayırmada yapısal bir avantaj sağlamalı. Sağlamazsa düz bir RNN yeterlidir.

ÇÜRÜTME KOŞULU
    Monte-Carlo ortalamasında entropi-Δt'nin Spearman'ı uniform-Δt'yi
    geçmiyorsa (veya proxy gürültüsüne tamamen dayanıksızsa), sürekli-zaman
    avantajı yok demektir.

YÖNTEM (modelsiz)
    Çok sayıda sentetik diyalog üret:
      - importance a_i ~ heavy-tailed (birkaç önemli, çok dolgu)
      - surprisal S_i = a_i + gürültü   (proxy)
      - girdi büyüklüğü ∝ a_i           (önemli turn daha güçlü sinyal)
    İki LTC koşusu (entropi-Δt vs uniform-Δt) → ‖Δx‖ profili → gerçek importance
    ile Spearman. Gürültü seviyesini de süpürüp dayanıklılığı ölç.
"""
import numpy as np
import matplotlib.pyplot as plt

from simulations.cfc_sims.sim_common import (
    LTCCell, surprisal_to_dt, spearman, savefig, verdict, DELTA_T_MIN, BETA,
)


def _one_dialogue(rng, dim, noise, cell):
    T = rng.integers(8, 16)
    # heavy-tailed importance: çoğu küçük, birkaçı büyük
    a = rng.exponential(0.4, size=T)
    a[rng.integers(0, T)] += rng.uniform(2.0, 4.0)  # en az bir güçlü turn
    a = np.clip(a, 0, None)

    # surprisal = importance + gürültü (proxy ilişkisi)
    surprisals = np.clip(a + rng.normal(0, noise, size=T), 0.05, None)

    # girdi büyüklüğü importance ile orantılı
    base = rng.normal(0, 1, size=(T, dim))
    inputs = base * (0.3 + a[:, None])

    dt_ent = surprisal_to_dt(surprisals, beta=BETA, dt_min=DELTA_T_MIN)
    dt_uni = np.full(T, dt_ent.mean())

    _, disp_ent = cell.run_sequence(inputs, dt_ent)
    _, disp_uni = cell.run_sequence(inputs, dt_uni)

    return spearman(a, disp_ent), spearman(a, disp_uni)


def run(seed: int = 0, n_dialogues: int = 200):
    rng = np.random.default_rng(seed)
    dim = 8
    cell = LTCCell(dim=dim, seed=seed)

    noise_levels = [0.1, 0.3, 0.6, 1.0, 1.5]
    mean_ent, mean_uni = [], []
    for noise in noise_levels:
        e, u = [], []
        for _ in range(n_dialogues):
            se, su = _one_dialogue(rng, dim, noise, cell)
            e.append(se); u.append(su)
        mean_ent.append(np.mean(e)); mean_uni.append(np.mean(u))
        print(f"  gürültü={noise:.1f} | Spearman  entropi-Δt={np.mean(e):+.3f}"
              f"  uniform-Δt={np.mean(u):+.3f}  (Δ={np.mean(e)-np.mean(u):+.3f})")

    mean_ent, mean_uni = np.array(mean_ent), np.array(mean_uni)

    # --- Görselleştirme ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(noise_levels, mean_ent, marker="o", color="#C44E52", label="entropi-Δt (liquid)")
    plt.plot(noise_levels, mean_uni, marker="s", color="#888888", label="uniform-Δt (discrete)")
    plt.xlabel("Proxy gürültü seviyesi  (surprisal ↔ importance)")
    plt.ylabel("Spearman( gerçek importance , ‖Δx‖ )")
    plt.title("Sürekli-zaman, önem sinyalini daha iyi koruyor mu?")
    plt.legend(); plt.grid(alpha=0.3)
    savefig("sim3_continuous_vs_discrete.png")

    avg_gain = float((mean_ent - mean_uni).mean())
    print(f"\n  Ortalama Spearman kazancı (entropi − uniform): {avg_gain:+.3f}")

    # --- Karar ---
    ok = (mean_ent > mean_uni).mean() >= 0.8 and avg_gain > 0.03
    verdict(ok,
            "Entropi-Δt, gürültü seviyelerinin çoğunda gerçek önemi uniform-Δt'den "
            "daha sadık sıralıyor. Sürekli-zaman avantajı kavramsal olarak destekleniyor.")
    return {"avg_gain": avg_gain, "mean_ent": mean_ent.tolist(),
            "mean_uni": mean_uni.tolist(), "passed": ok}


if __name__ == "__main__":
    run()
