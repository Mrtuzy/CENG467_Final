"""
S5 — Δt eşleme fonksiyonunu simülasyon seçer (config tasarımı)
=============================================================
SORU
    Projenin Δt = Δt_min + β·S (DOĞRUSAL, sınırsız) eşlemesi en iyisi mi?
    Gerçek surprisal AĞIR KUYRUKLUDUR (DistilGPT-2 nadiren çok yüksek değer verir).
    Tek bir aykırı-surprisal turn doğrusal eşlemede dev Δt alır → kendi bağlanmasını
    doyurur VE recency yoluyla sonraki turn'lerin etkisini ezer. Bu, kurtarmayı
    bozabilir.

KARŞILAŞTIRILAN EŞLEMELER (core.MAPPINGS)
    • linear : Δt = Δt_min + β·S                 (mevcut)
    • bounded: Δt = Δt_min + β·s0·tanh(S/s0)     (doygun, aykırıya dayanıklı)
    • rank   : Δt = Δt_min + β·rank(S)/N          (ölçek-bağımsız, sıra-temelli)

YÖNTEM
    Gerçekçi ρ'de iki rejim: (a) normal surprisal, (b) aykırı enjekte edilmiş.
    Kurtarma = Spearman(influence, u). Dayanıklılık = aykırı varken kayıp.

ÇIKTI → TASARIM
    En yüksek + en dayanıklı eşleme config.py için önerilir (build_inputs.py'deki
    surprisal→Δt adımını değiştirir).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import (MAPPINGS, readout_signal, make_surprisal, heavy_tailed_importance,
                  spearman, bootstrap_ci, savefig, verdict, READOUT_NOISE,
                  C_LIQUID, C_BASE, C_ALT)

PROXY_RHO = 0.6
COLORS = {"linear": C_BASE, "bounded": C_ALT, "rank": C_LIQUID}


def _eval(mapping_fn, seed, outlier, n_dialogues=200):
    # Sabit hesap bütçesi: Σ Δt sabit → eşlemenin ÖLÇEĞİ değil ŞEKLİ ölçülür.
    # Aykırı surprisal'lı bir turn doğrusal eşlemede bütçenin çoğunu kapar ve
    # diğer (önemli) turn'leri aç bırakır; sıra/sınırlı eşleme bunu önler.
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_dialogues):
        T = rng.integers(8, 18)
        u = heavy_tailed_importance(T, rng)
        S = make_surprisal(u, PROXY_RHO, rng)
        if outlier:
            S = S.copy(); S[rng.integers(0, T)] *= rng.uniform(6, 12)  # aykırı surprisal
        dt = mapping_fn(S)
        sig = readout_signal(dt, noise=READOUT_NOISE, rng=rng, budget=float(T))
        scores.append(spearman(sig, u))
    return np.mean(scores)


def run(seed: int = 0, n_seeds: int = 6):
    res = {}
    for name, fn in MAPPINGS.items():
        clean = np.array([_eval(fn, seed + s, outlier=False) for s in range(n_seeds)])
        noisy = np.array([_eval(fn, seed + s, outlier=True) for s in range(n_seeds)])
        mc, lc, hc = bootstrap_ci(clean)
        mn, ln, hn = bootstrap_ci(noisy)
        res[name] = dict(clean=mc, clean_lo=lc, clean_hi=hc,
                         noisy=mn, noisy_lo=ln, noisy_hi=hn, drop=mc - mn)
        print(f"  {name:8s} | normal={mc:+.3f}  aykırılı={mn:+.3f}  "
              f"dayanıklılık-kaybı={mc-mn:+.3f}")

    # En iyi: aykırılı rejimdeki kurtarmaya göre (gerçekçi en kötü durum)
    best = max(res, key=lambda k: res[k]["noisy"])
    print(f"\n  Öneri (aykırıya dayanıklı): '{best}' eşlemesi.")

    # --- Görselleştirme: gruplu bar + hata çubukları ---
    names = list(MAPPINGS.keys())
    x = np.arange(len(names)); w = 0.36
    clean_m = [res[n]["clean"] for n in names]
    noisy_m = [res[n]["noisy"] for n in names]
    clean_e = [[res[n]["clean"] - res[n]["clean_lo"] for n in names],
               [res[n]["clean_hi"] - res[n]["clean"] for n in names]]
    noisy_e = [[res[n]["noisy"] - res[n]["noisy_lo"] for n in names],
               [res[n]["noisy_hi"] - res[n]["noisy"] for n in names]]

    plt.figure(figsize=(8.5, 5))
    plt.bar(x - w/2, clean_m, w, yerr=clean_e, capsize=4, color=C_BASE,
            label="normal surprisal", alpha=0.85)
    plt.bar(x + w/2, noisy_m, w, yerr=noisy_e, capsize=4, color=C_LIQUID,
            label="outlier surprisal injected", alpha=0.85)
    plt.xticks(x, names)
    plt.ylabel("importance recovery  Spearman(signal, u)")
    plt.title("S5 — Which Δt mapping is robust to outlier surprisal?")
    plt.axhline(0, color="k", lw=0.6)
    plt.legend(); plt.grid(alpha=0.3, axis="y")
    savefig("s5_mapping_function_design.png")

    ok = res[best]["noisy"] >= res["linear"]["noisy"]
    verdict(ok, f"En dayanıklı eşleme '{best}'. Doğrusal eşleme aykırı surprisal'da "
                f"{res['linear']['drop']:+.3f} kaybediyor; '{best}' "
                f"{res[best]['drop']:+.3f}. Öneri: build_inputs.py'de surprisal→Δt "
                f"adımını '{best}' yap (özellikle ham DistilGPT-2 surprisal'ı kullanılıyorsa).")
    return {"results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
            "recommended": best, "passed": ok}


if __name__ == "__main__":
    run()
