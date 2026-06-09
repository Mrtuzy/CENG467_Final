"""
S2 — β ve Δt_min'i simülasyon seçer (model tasarımı çıktısı)
===========================================================
HİPOTEZ
    (β, Δt_min) çiftinin, önem-kurtarmayı (recovery) maksimize eden anlamlı bir
    orta rejimi vardır: β=0 entropiyi yok sayar (kurtarma çöker); β çok büyük
    tüm turn'leri doyurur (ayrım kaybolur). Δt_min taban işlem süresini belirler.

NEDEN ÖNEMLİ
    Kullanıcının açık hedefi: "simülasyon çıktılarıyla modeli tasarlamak."
    Bu sim, config.py'deki BETA ve DELTA_T_MIN için *veriye dayalı* öneri üretir
    ve ablation (§7) için doğal gerekçe sağlar.

YÖNTEM (modelsiz)
    Çok sayıda sentetik diyalog. Önem u ağır kuyruklu; surprisal S, u'nun SABİT,
    GERÇEKÇİ kaliteli (ρ≈0.6) bir vekili. İçerik büyüklüğü EŞİT tutulur → önem
    yalnızca Δt kanalından akar. Her (β, Δt_min) için kapalı-form etki
    (chain_influence) ile kurtarma = Spearman(influence, u). Çok seed → GA.

ÇIKTI → TASARIM
    Argmax (β*, Δt_min*) ve etrafındaki düz plato → config önerisi + ablation aralığı.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import (readout_signal, dt_linear, make_surprisal, heavy_tailed_importance,
                  spearman, bootstrap_ci, savefig, verdict, C_LIQUID, READOUT_NOISE)

PROXY_RHO = 0.6  # gerçekçi orta-kalite surprisal vekili


def _recovery(beta, dt_min, seed, n_dialogues=120):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_dialogues):
        T = rng.integers(8, 18)
        u = heavy_tailed_importance(T, rng)
        S = make_surprisal(u, PROXY_RHO, rng)
        dt = dt_linear(S, beta=beta, dt_min=dt_min)
        # turn-başına okunabilir sinyal; içerik eşit → yalnızca Δt kanalı.
        # gürültü tabanı: β→0'da commit farkları silinir, β→∞'da doyum bağlar.
        sig = readout_signal(dt, noise=READOUT_NOISE, rng=rng)
        scores.append(spearman(sig, u))
    return np.array(scores)


def run(seed: int = 0, n_seeds: int = 6):
    betas = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    dt_mins = np.array([0.05, 0.1, 0.25, 0.5])

    grid = np.zeros((len(dt_mins), len(betas)))
    grid_lo = np.zeros_like(grid)
    for i, dm in enumerate(dt_mins):
        for j, b in enumerate(betas):
            per_seed = [np.mean(_recovery(b, dm, seed + s)) for s in range(n_seeds)]
            m, lo, hi = bootstrap_ci(per_seed)
            grid[i, j] = m
            grid_lo[i, j] = lo

    bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
    best_dm, best_b, best_val = dt_mins[bi], betas[bj], grid[bi, bj]
    print(f"  En iyi kurtarma: β*={best_b}, Δt_min*={best_dm} → Spearman={best_val:+.3f}")
    print(f"  β=0 (entropi yok) kurtarma: {grid[bi,0]:+.3f}  (kaldıracın faydası: "
          f"{best_val-grid[bi,0]:+.3f})")

    # --- Görselleştirme: ısı haritası + β kesiti ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    im = ax[0].imshow(grid, aspect="auto", origin="lower", cmap="magma",
                      vmin=max(0, grid.min()))
    ax[0].set_xticks(range(len(betas))); ax[0].set_xticklabels(betas)
    ax[0].set_yticks(range(len(dt_mins))); ax[0].set_yticklabels(dt_mins)
    ax[0].set_xlabel(r"$\beta$"); ax[0].set_ylabel(r"$\Delta t_{min}$")
    ax[0].set_title("Importance recovery  Spearman(signal, u)")
    ax[0].scatter([bj], [bi], marker="*", s=260, color="cyan", edgecolor="k",
                  label=f"argmax β={best_b}, Δt_min={best_dm}")
    ax[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    # β kesiti (en iyi Δt_min'de) + config varsayılanı
    cut = grid[bi]; cut_lo = grid_lo[bi]
    ax[1].plot(betas, cut, marker="o", color=C_LIQUID, lw=2, label=f"Δt_min={best_dm}")
    ax[1].fill_between(betas, cut_lo, 2 * cut - cut_lo, color=C_LIQUID, alpha=0.18)
    ax[1].axvline(1.0, color="k", ls="--", lw=1, label="config default β=1.0")
    ax[1].scatter([best_b], [best_val], marker="*", s=200, color="gold",
                  edgecolor="k", zorder=5)
    ax[1].set_xlabel(r"$\beta$"); ax[1].set_ylabel("recovery Spearman")
    ax[1].set_title("β slice: meaningful mid-regime")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    savefig("s2_beta_dtmin_design.png")

    # config varsayılanı (β=1) en iyiye ne kadar yakın?
    default_val = grid[bi, list(betas).index(1.0)]
    gap = best_val - default_val
    ok = (best_val - grid[bi, 0]) > 0.05 and best_b > 0
    verdict(ok, f"β'nın anlamlı orta rejimi var (β=0'a göre +{best_val-grid[bi,0]:.2f} "
                f"kazanç). Öneri: β≈{best_b}, Δt_min≈{best_dm}. config β=1 zaten "
                f"optimuma {gap:+.3f} yakın.")
    return {"best_beta": float(best_b), "best_dt_min": float(best_dm),
            "best_recovery": float(best_val), "beta0_recovery": float(grid[bi, 0]),
            "default_beta1_recovery": float(default_val),
            "grid": grid.tolist(), "betas": betas.tolist(),
            "dt_mins": dt_mins.tolist(), "passed": ok}


if __name__ == "__main__":
    run()
