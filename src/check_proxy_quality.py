"""
Eğitim-Öncesi Go/No-Go: surprisal ↔ teacher-önem korelasyonu
============================================================
simulations/cfc_proof/S3, tüm yöntemin tek kritik varsayımını ortaya koydu:
*surprisal, gerçek önemin yeterince iyi bir vekili değilse entropi-Δt hiçbir şey
kazandırmaz.* Bu script, o varsayımı CfC'yi EĞİTMEDEN ÖNCE gerçek veride ölçer.

Δt = f(surprisal) eşlemesi (linear veya rank) surprisal'da MONOTON olduğundan,
Spearman(Δt, teacher_target) = Spearman(surprisal, teacher_target). Bu yüzden
build_inputs.py'nin ürettiği delta_t.pt ile teacher_targets.pt yeterlidir.

Kullanım:
    python src/check_proxy_quality.py

Çıktı: bağlam-içi Spearman dağılımı + bootstrap %95 GA + go/no-go kararı.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, FIGURES_DIR, DT_MAPPING

# S3'ten eşikler (simulations/cfc_proof/RESULTS.md):
RHO_BEAT_UNIFORM = 0.1   # bunun altında entropi-Δt uniform'dan farksız → NO-GO
RHO_BEAT_COSINE = 0.5    # bunun üstünde cosine baseline'ı da geçer → güçlü GO


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def _bootstrap_ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def check_proxy_quality():
    dt_path = os.path.join(OUTPUT_DIR, "delta_t.pt")
    tgt_path = os.path.join(OUTPUT_DIR, "teacher_targets.pt")
    for p in (dt_path, tgt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} bulunamadı. Önce build_inputs.py ve teacher_labeling.py çalışmalı.")

    delta_ts = torch.load(dt_path, map_location="cpu", weights_only=False)
    targets = torch.load(tgt_path, map_location="cpu", weights_only=False)

    per_ctx = []
    skipped = 0
    for dt, tg in zip(delta_ts, targets):
        dt = np.asarray(dt, float).ravel()
        tg = np.asarray(tg, float).ravel()
        if dt.size != tg.size or dt.size < 3:
            skipped += 1
            continue
        rho = _spearman(dt, tg)
        if rho is None:
            skipped += 1
            continue
        per_ctx.append(rho)

    if not per_ctx:
        print("Yeterli (≥3 turn'lü) bağlam yok; go/no-go hesaplanamadı.")
        return

    per_ctx = np.array(per_ctx)
    mean, lo, hi = _bootstrap_ci(per_ctx)
    median = float(np.median(per_ctx))
    frac_pos = float((per_ctx > 0).mean())

    print("=" * 64)
    print("EĞİTİM-ÖNCESİ GO/NO-GO — surprisal ↔ teacher-önem (S3)")
    print("=" * 64)
    print(f"  Δt eşlemesi (config.DT_MAPPING): {DT_MAPPING}")
    print(f"  Değerlendirilen bağlam: {per_ctx.size}  (atlanan <3 turn: {skipped})")
    print(f"  Bağlam-içi Spearman(surprisal, önem):")
    print(f"     ortalama = {mean:+.3f}   %95 GA = [{lo:+.3f}, {hi:+.3f}]")
    print(f"     medyan   = {median:+.3f}   pozitif bağlam oranı = {frac_pos:.0%}")

    if mean >= RHO_BEAT_COSINE:
        verdict = "GÜÇLÜ GO — entropi-Δt cosine baseline'ı da geçmeli; CfC eğitimi değer."
    elif mean >= RHO_BEAT_UNIFORM:
        verdict = ("GO — entropi-Δt uniform-Δt'yi geçmeli ama cosine'e yakın; "
                   "CfC'yi eğit ve cosine baseline ile karşılaştır.")
    else:
        verdict = ("NO-GO — surprisal önemi yeterince vekillemiyor (ρ<%.2f). "
                   "Entropi kaldıracı muhtemelen kazandırmaz; cosine baseline "
                   "yeterli olabilir veya proxy modeli/eşlemeyi gözden geçir." % RHO_BEAT_UNIFORM)
    print(f"\n  KARAR: {verdict}")
    print("=" * 64)

    # --- histogram figürü ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.figure(figsize=(8, 4.4))
        plt.hist(per_ctx, bins=30, color="#C44E52", alpha=0.85, edgecolor="white")
        plt.axvline(mean, color="k", lw=2, label=f"mean={mean:+.3f}")
        plt.axvline(RHO_BEAT_UNIFORM, color="#888888", ls="--", lw=1.5,
                    label=f"go/no-go eşiği ρ*={RHO_BEAT_UNIFORM}")
        plt.axvline(RHO_BEAT_COSINE, color="#4C72B0", ls=":", lw=1.5,
                    label=f"cosine'i geçme ρ={RHO_BEAT_COSINE}")
        plt.xlabel("per-context Spearman(surprisal, teacher importance)")
        plt.ylabel("number of contexts")
        plt.title("Pre-training go/no-go: is surprisal a good proxy for importance?")
        plt.legend(); plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "proxy_quality_gonogo.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"  📊 figür → {out}")
    except Exception as e:
        print(f"  (figür oluşturulamadı: {e})")

    # makine-okunur özet
    summary = {"mapping": DT_MAPPING, "n_contexts": int(per_ctx.size),
               "mean": mean, "ci95": [lo, hi], "median": median,
               "frac_positive": frac_pos,
               "rho_star_uniform": RHO_BEAT_UNIFORM, "rho_cosine": RHO_BEAT_COSINE}
    with open(os.path.join(OUTPUT_DIR, "proxy_quality.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    check_proxy_quality()
