"""
run_all.py — CfC kanıt-suite'ini baştan sona çalıştır
=====================================================
Her sim'i çalıştırır, figürleri figures/ altına yazar, results.json üretir
ve tek bir özet tablo basar. Model yüklemez, eğitim yapmaz.

    python run_all.py
"""
import os, sys, json, importlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _json_default(o):
    """numpy skalerlerini/bool'larını/dizilerini JSON'a çevir."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

SIMS = [
    ("S0", "s0_substrate_faithfulness", "analitik LTC ↔ gerçek CfC sadakati"),
    ("S1", "s1_mechanism_analytic", "entropi→Δt mekanizması (analitik, tautoloji değil)"),
    ("S2", "s2_beta_dtmin_design", "β & Δt_min önerisi (tasarım)"),
    ("S3", "s3_proxy_quality_limit", "surprisal↔önem kritik ρ* (go/no-go)"),
    ("S4", "s4_loo_kl_redundancy", "leave-one-out KL = marjinal cevap-gerekliliği"),
    ("S5", "s5_mapping_function_design", "Δt eşleme fonksiyonu önerisi"),
    ("S6", "s6_endtoend_pruning", "sadakat–sıkıştırma + τ önerisi"),
]


def main():
    results = {}
    for tag, mod_name, desc in SIMS:
        print("=" * 72)
        print(f"{tag} — {desc}")
        print("=" * 72)
        mod = importlib.import_module(mod_name)
        results[tag] = mod.run()
        print()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    print("=" * 72)
    print("ÖZET")
    print("=" * 72)
    print(f"{'Sim':<5}{'Sonuç':<12}{'Açıklama'}")
    print("-" * 72)
    for tag, _, desc in SIMS:
        ok = results[tag].get("passed", False)
        print(f"{tag:<5}{'✓ TUTARLI' if ok else '✗ TUTARSIZ':<12}{desc}")
    print("-" * 72)

    # tasarım çıktıları (modeli tasarlamak için)
    print("\nMODEL TASARIM ÇIKTILARI:")
    if "S2" in results:
        print(f"  • β*           = {results['S2'].get('best_beta')}  "
              f"(config.BETA, varsayılan 1.0)")
        print(f"  • Δt_min*      = {results['S2'].get('best_dt_min')}  "
              f"(config.DELTA_T_MIN, varsayılan 0.1)")
    if "S3" in results:
        print(f"  • kritik ρ*    = {results['S3'].get('rho_star')}  "
              f"(QReCC'de corr(surprisal, teacher-önem) bunu aşmalı — go/no-go)")
    if "S5" in results:
        print(f"  • Δt eşlemesi  = '{results['S5'].get('recommended')}'  "
              f"(build_inputs.py surprisal→Δt adımı)")
    if "S6" in results:
        print(f"  • τ*           = {results['S6'].get('tau_star')}  "
              f"(config.TAU, varsayılan 0.5)")
    print(f"\n✓ results.json → {out}")


if __name__ == "__main__":
    main()
