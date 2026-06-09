"""
S3 — Bağlayıcı kısıt: surprisal ↔ ÖNEM korelasyonu (dürüst sınır)
================================================================
ASIL SORU
    Tüm yöntemin gerçek riski şudur: surprisal, gerçek ÖNEM'in iyi bir vekili
    DEĞİLSE entropi-Δt hiçbir şey kazandırmaz. Bu, projenin başarısının
    bağlandığı tek kritik varsayımdır. Onu doğrudan ve acımasızca test ederiz.

YÖNTEM (modelsiz, kanal-ayrıştırılmış)
    Latent önem u ile surprisal S arasındaki korelasyonu ρ ∈ [0,1] olarak
    SÜPÜRÜRÜZ. İçerik büyüklüğü EŞİT (input kanalı kapalı) → önem sinyali
    yalnızca Δt'den gelebilir. Karşılaştırma:
      • entropi-Δt  (liquid): Δt ∝ surprisal
      • uniform-Δt  (discrete-RNN benzeri): Δt sabit → kurtarma ≈ 0 (referans taban)
      • cosine-vekili baseline: bağımsız, ρ_cos kaliteli ikinci bir vekil
    Kurtarma = Spearman(influence, u). Çok seed → GA. Kritik ρ* (entropi-Δt'nin
    uniform'u anlamlı geçtiği eşik) bulunur.

ÇIKTI → TASARIM  (go / no-go)
    ρ* eşiği, modeli EĞİTMEDEN ÖNCE gerçek QReCC üzerinde ölçülmesi gereken
    somut bir koşul verir: corr(surprisal, teacher-önem) > ρ* değilse, entropi
    kaldıracı işe yaramaz ve cosine baseline yeterlidir. Bu, dürüst bir kırılma
    noktasıdır — "her şey çalışıyor" demekten bilimsel olarak daha güçlüdür.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import (readout_signal, dt_linear, make_surprisal, heavy_tailed_importance,
                  spearman, bootstrap_ci, savefig, verdict, READOUT_NOISE,
                  C_LIQUID, C_BASE, C_ALT)

RHO_COS = 0.45  # cosine-benzeri baseline'ın önemle (sabit) vekil kalitesi


def _sweep_point(rho, seed, n_dialogues=200):
    rng = np.random.default_rng(seed)
    ent, uni, cos = [], [], []
    for _ in range(n_dialogues):
        T = rng.integers(8, 18)
        u = heavy_tailed_importance(T, rng)
        S = make_surprisal(u, rho, rng)            # surprisal: ρ kaliteli vekil
        Scos = make_surprisal(u, RHO_COS, rng)     # cosine: sabit kaliteli vekil

        dt_e = dt_linear(S)
        dt_u = np.full(T, dt_e.mean())             # aynı toplam zaman, uniform
        ent.append(spearman(readout_signal(dt_e, noise=READOUT_NOISE, rng=rng), u))
        uni.append(spearman(readout_signal(dt_u, noise=READOUT_NOISE, rng=rng), u))
        # cosine baseline doğrudan vekili önem skoru olarak kullanır (Δt yok)
        cos.append(spearman(Scos, u))
    return np.mean(ent), np.mean(uni), np.mean(cos)


def run(seed: int = 0, n_seeds: int = 6):
    rhos = np.linspace(0.0, 1.0, 11)
    E, U, COS = [], [], []
    Elo, Ehi = [], []
    for rho in rhos:
        per = np.array([_sweep_point(rho, seed + s) for s in range(n_seeds)])
        e, l, h = bootstrap_ci(per[:, 0]); E.append(e); Elo.append(l); Ehi.append(h)
        U.append(per[:, 1].mean()); COS.append(per[:, 2].mean())
        print(f"  ρ(S,u)={rho:.1f} | entropi-Δt={e:+.3f}  uniform={U[-1]:+.3f}  "
              f"cosine={COS[-1]:+.3f}")
    E, U, COS = map(np.array, (E, U, COS))

    # Kritik ρ*: entropi-Δt'nin uniform'u +0.03 marjla geçtiği en küçük ρ
    over = (E - U) > 0.03
    rho_star = float(rhos[np.argmax(over)]) if over.any() else float("nan")
    # entropi-Δt'nin cosine baseline'ı geçtiği eşik
    over_cos = E > COS
    rho_cos_cross = float(rhos[np.argmax(over_cos)]) if over_cos.any() else float("nan")
    print(f"\n  Kritik ρ* (entropi > uniform): {rho_star}")
    print(f"  entropi > cosine-baseline eşiği: {rho_cos_cross}")

    # --- Görselleştirme ---
    plt.figure(figsize=(8.5, 5))
    plt.plot(rhos, E, marker="o", color=C_LIQUID, lw=2.4, label="entropy-Δt (liquid)")
    plt.fill_between(rhos, Elo, Ehi, color=C_LIQUID, alpha=0.18)
    plt.plot(rhos, U, marker="s", color=C_BASE, lw=2, label="uniform-Δt (discrete-RNN)")
    plt.plot(rhos, COS, marker="^", color=C_ALT, lw=2, ls="--",
             label=f"cosine baseline (ρ_cos={RHO_COS})")
    if not np.isnan(rho_star):
        plt.axvline(rho_star, color="k", ls=":", lw=1.2)
        plt.text(rho_star + 0.01, 0.02, f"ρ* ≈ {rho_star:.1f}\n(go/no-go threshold)",
                 fontsize=9)
    plt.xlabel(r"surprisal ↔ IMPORTANCE correlation  $\rho$  (MEASURE on real QReCC)")
    plt.ylabel("importance recovery  Spearman(signal, u)")
    plt.title("S3 — The method helps only if surprisal proxies importance well enough")
    plt.legend(loc="upper left"); plt.grid(alpha=0.3)
    savefig("s3_proxy_quality_limit.png")

    # Sim "geçti" demek = sınırı düzgün ortaya koyuyor: yüksek ρ'da kazanç, düşükte yok
    ok = (E[-1] - U[-1]) > 0.1 and (E[0] - U[0]) < 0.05 and not np.isnan(rho_star)
    verdict(ok, f"Entropi-Δt'nin avantajı ρ ile DÜZGÜN ortaya çıkıyor: ρ<{rho_star:.1f} "
                f"iken uniform'dan farksız, yüksek ρ'da belirgin. Modeli eğitmeden "
                f"önce QReCC'de ρ>ρ* doğrulanmalı; aksi halde cosine baseline yeterli.")
    return {"rho_star": rho_star, "rho_cos_cross": rho_cos_cross,
            "rhos": rhos.tolist(), "entropy": E.tolist(), "uniform": U.tolist(),
            "cosine": COS.tolist(), "passed": ok}


if __name__ == "__main__":
    run()
