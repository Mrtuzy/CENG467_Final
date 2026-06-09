"""
S6 — Uçtan uca pruning vekili (sunum başlığı + τ seçimi)
=======================================================
SORU
    Önceki sim'ler mekanizmayı parça parça doğruladı. Bu sim, izleyicinin
    GERÇEKTEN umursadığı düzeyde bütünü gösterir: "ne kadar bağlam budarsak
    cevabı korumak için gereken turn'leri ne kadar tutarız?" — modelsiz vekil.

KURULUM (modelsiz)
    Her bağlamda T turn; cevabı üretmek için GEREKLİ küçük bir alt-küme (needed
    set = en yüksek önemli m turn). Bir budayıcı, turn'leri skorlayıp eşik τ
    altındakileri atar. Karşılaştırılan skorlayıcılar:
      • entropi-Δt (liquid): chain_influence(Δt∝surprisal)
      • uniform-Δt: chain_influence(sabit Δt)
      • cosine baseline: sorguya benzerlik vekili (sabit kalite)
      • random
    τ süpürülür → (sıkıştırma, sadakat) eğrisi. Sadakat = gerekli turn'lerin
    geri-çağrımı (recall); sıkıştırma = budanan turn payı.

ÇIKTI → TASARIM
    • Önerilen τ (config.TAU): hedef sadakatte (örn. %90 recall) en yüksek sıkıştırma.
    • Tek başlık figürü: "%X sıkıştırmada entropi-CfC gerekli turn'lerin %Y'sini
      tutuyor; baseline'lar %Z." Bu, raporun/sunumun merkez görselidir.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import (readout_signal, dt_linear, make_surprisal, heavy_tailed_importance,
                  bootstrap_ci, savefig, verdict, READOUT_NOISE, C_LIQUID, C_BASE, C_ALT)

PROXY_RHO = 0.6
RHO_COS = 0.45
TARGET_FIDELITY = 0.90  # hedef: gerekli turn'lerin %90'ını koru


def _scores(rng):
    T = rng.integers(10, 20)
    u = heavy_tailed_importance(T, rng)
    m = max(2, T // 5)
    needed = set(np.argsort(u)[-m:])          # cevabı üretmek için gerekli turn'ler

    S = make_surprisal(u, PROXY_RHO, rng)
    dt_e = dt_linear(S)
    infl_ent = readout_signal(dt_e, noise=READOUT_NOISE, rng=rng)
    infl_uni = readout_signal(np.full(T, dt_e.mean()), noise=READOUT_NOISE, rng=rng)
    cos = make_surprisal(u, RHO_COS, rng)
    rnd = rng.random(T)
    return needed, T, dict(entropy=infl_ent, uniform=infl_uni, cosine=cos, random=rnd)


def _curve(score, needed, T, thresholds):
    """Skoru normalize et, τ süpür → (sıkıştırma, sadakat-recall) listeleri."""
    s = (score - score.min()) / (np.ptp(score) + 1e-12)
    comp, fid = [], []
    nd = np.array(sorted(needed))
    for t in thresholds:
        keep = s >= t
        comp.append(1.0 - keep.mean())                  # budanan oran
        fid.append(keep[nd].mean() if nd.size else 1.0)  # gerekli turn recall
    return np.array(comp), np.array(fid)


def run(seed: int = 0, n_seeds: int = 6, n_ctx=400):
    thr = np.linspace(0, 1, 41)
    methods = ["entropy", "uniform", "cosine", "random"]
    # her metot için τ-ızgarasında ortalama (sıkıştırma, sadakat)
    agg = {mname: {"comp": [], "fid": []} for mname in methods}
    for s in range(n_seeds):
        rng = np.random.default_rng(seed + s)
        per = {mname: {"comp": np.zeros_like(thr), "fid": np.zeros_like(thr)} for mname in methods}
        for _ in range(n_ctx):
            needed, T, sc = _scores(rng)
            for mname in methods:
                c, f = _curve(sc[mname], needed, T, thr)
                per[mname]["comp"] += c; per[mname]["fid"] += f
        for mname in methods:
            agg[mname]["comp"].append(per[mname]["comp"] / n_ctx)
            agg[mname]["fid"].append(per[mname]["fid"] / n_ctx)

    summary = {}
    print(f"  Hedef sadakat (gerekli turn recall) = {TARGET_FIDELITY:.0%}")
    plt.figure(figsize=(8.8, 5.2))
    for mname in methods:
        comp = np.mean(agg[mname]["comp"], axis=0)
        fid = np.mean(agg[mname]["fid"], axis=0)
        col = {"entropy": C_LIQUID, "uniform": C_BASE,
               "cosine": C_ALT, "random": "#BBBBBB"}[mname]
        lw = 2.6 if mname == "entropy" else 1.8
        plt.plot(comp, fid, marker="o", ms=3, color=col, lw=lw,
                 label=mname + (" (liquid)" if mname == "entropy" else ""))
        # hedef sadakatte ulaşılabilen maksimum sıkıştırma
        ok_mask = fid >= TARGET_FIDELITY
        max_comp = float(comp[ok_mask].max()) if ok_mask.any() else 0.0
        # ortak sıkıştırma noktasında (50%) sadakat
        idx50 = np.argmin(np.abs(comp - 0.5))
        summary[mname] = {"max_comp_at_target": max_comp, "fid_at_50pct_comp": float(fid[idx50])}
        print(f"  {mname:8s} | {TARGET_FIDELITY:.0%} sadakatte max sıkıştırma="
              f"{max_comp:.2f} | %50 sıkıştırmada sadakat={fid[idx50]:.2f}")

    plt.axhline(TARGET_FIDELITY, color="k", ls="--", lw=1,
                label=f"target fidelity {TARGET_FIDELITY:.0%}")
    plt.xlabel("compression (fraction of turns pruned)")
    plt.ylabel("fidelity (needed-turn recall)")
    plt.title("S6 — Fidelity–Compression: entropy-CfC proxy vs baselines")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    savefig("s6_endtoend_pruning.png")

    # Önerilen τ: entropi metodu, hedef sadakati hâlâ tutan en yüksek τ
    comp_e = np.mean(agg["entropy"]["comp"], axis=0)
    fid_e = np.mean(agg["entropy"]["fid"], axis=0)
    ok_mask = fid_e >= TARGET_FIDELITY
    tau_star = float(thr[ok_mask].max()) if ok_mask.any() else float("nan")
    print(f"\n  Önerilen τ (config.TAU): ~{tau_star:.2f}  "
          f"(entropi metodu {TARGET_FIDELITY:.0%} sadakati koruyan en yüksek eşik)")

    ent_c = summary["entropy"]["max_comp_at_target"]
    uni_c = summary["uniform"]["max_comp_at_target"]
    ok = ent_c >= uni_c and ent_c >= summary["random"]["max_comp_at_target"]
    verdict(ok, f"Entropi-CfC vekili, {TARGET_FIDELITY:.0%} sadakatte {ent_c:.0%} "
                f"sıkıştırmaya izin veriyor; uniform {uni_c:.0%}, cosine "
                f"{summary['cosine']['max_comp_at_target']:.0%}. Önerilen τ≈{tau_star:.2f}. "
                "Bu, raporun başlık sonucudur.")
    return {"summary": summary, "tau_star": tau_star,
            "target_fidelity": TARGET_FIDELITY, "passed": ok}


if __name__ == "__main__":
    run()
