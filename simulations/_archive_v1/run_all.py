"""
Tüm kavram-doğrulama simülasyonlarını sırayla çalıştırır ve özet basar.

    python simulations/run_all.py

Her simülasyon modelsizdir (numpy + matplotlib), bir hipotezi test eder,
bir figür kaydeder (simulations/figures/) ve TUTARLI/TUTARSIZ kararı verir.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sim1_entropy_dt_mapping as s1
import sim2_beta_sensitivity as s2
import sim3_continuous_vs_discrete as s3
import sim4_loo_kl_premise as s4


def main():
    sims = [
        ("Sim1 · Entropi→Δt eşlemesi", s1.run),
        ("Sim2 · β hassasiyeti",        s2.run),
        ("Sim3 · Sürekli vs ayrık zaman", s3.run),
        ("Sim4 · Leave-one-out KL önermesi", s4.run),
    ]
    results = {}
    for title, fn in sims:
        print("=" * 70)
        print(title)
        print("=" * 70)
        results[title] = fn()

    print("=" * 70)
    print("ÖZET")
    print("=" * 70)
    for title, r in results.items():
        tag = "✓ TUTARLI" if r.get("passed") else "✗ TUTARSIZ"
        print(f"  {tag:14} {title}")

    def _clean(v):
        import numpy as np
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump({k: {kk: _clean(vv) for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=2, ensure_ascii=False)
    print(f"\n✓ Sonuçlar → {out}")

    all_ok = all(r.get("passed") for r in results.values())
    print(f"\n{'TÜM HİPOTEZLER TUTARLI ✓' if all_ok else 'BAZI HİPOTEZLER TUTARSIZ ✗'}")


if __name__ == "__main__":
    main()
