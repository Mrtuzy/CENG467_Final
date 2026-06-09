"""
Simülasyon 1 — Entropi→Δt eşlemesi mekanizması sağlam mı?
=========================================================
HİPOTEZ
    Δt_i = Δt_min + β·S_i haritalaması, yüksek-surprisal (entropili) turn'lere
    daha uzun "işlem zamanı" vererek bu turn'lerin sinyaline daha güçlü
    BAĞLANMAYI (denge-noktasına yaklaşma oranı φ) sağlar; düşük-entropili dolgu
    turn'ler için bu bağlanma zayıf kalır.

NEDEN ÖNEMLİ
    Bütün projenin önermesi, surprisal'ı bir "önem proxy"si olarak Δt'ye
    haritalamanın modele yararlı bir endüktif önyargı verdiği. Eğer Δt'yi
    değiştirmek bağlanmayı değiştirmiyorsa, bu haritalama anlamsızdır.

ÖLÇÜT — neden ham ‖Δx‖ değil?
    Ham durum hareketi, Δt etkisini turn'ün denge-noktası uzaklığıyla
    karıştırır (input nereye çekiyorsa hareket ona bağlı). Bu yüzden denge
    uzaklığına normalize edilmiş "bağlanma oranı" φ ∈ [0,1] kullanırız:
        φ = ‖x_sonra − x_önce‖ / ‖x* − x_önce‖
    φ, yapısal olarak Δt ile monoton artar; saf mekanizmayı izole eder.

ÇÜRÜTME KOŞULU
    Entropi-Δt rejiminde φ ile surprisal korelasyonu ≈ 0 ise, VEYA uniform-Δt
    ile aynıysa, mekanizma işe yaramıyor demektir.
"""
import numpy as np
import matplotlib.pyplot as plt

from simulations.cfc_sims.sim_common import (
    LTCCell, surprisal_to_dt, pearson, savefig, verdict, BETA, DELTA_T_MIN,
)


def _commitment_profile(cell, inputs, dts):
    """Bir dizi için turn başına bağlanma oranı φ_i (durumu zincirleyerek)."""
    x = np.zeros(cell.dim)
    phis = []
    for I, dt in zip(inputs, dts):
        phi, x = cell.commitment_fraction(x, I, dt)
        phis.append(phi)
    return np.array(phis)


def run(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, dim = 12, 8

    inputs = rng.normal(0, 1, size=(T, dim))
    surprisals = rng.uniform(0.2, 1.0, size=T)
    spike_idx = [2, 7, 10]
    surprisals[spike_idx] = rng.uniform(4.0, 5.5, size=len(spike_idx))

    cell = LTCCell(dim=dim, seed=seed)

    dt_entropy = surprisal_to_dt(surprisals, beta=BETA, dt_min=DELTA_T_MIN)
    dt_uniform = np.full(T, dt_entropy.mean())  # toplam zaman eşit

    phi_entropy = _commitment_profile(cell, inputs, dt_entropy)
    phi_uniform = _commitment_profile(cell, inputs, dt_uniform)

    r_entropy = pearson(surprisals, phi_entropy)
    r_uniform = pearson(surprisals, phi_uniform)

    print("Surprisal ↔ bağlanma oranı φ korelasyonu:")
    print(f"  entropi-Δt : r = {r_entropy:+.3f}")
    print(f"  uniform-Δt : r = {r_uniform:+.3f}")
    print(f"  Spike turn'lerin ort. φ (entropi): {phi_entropy[spike_idx].mean():.3f}")
    print(f"  Dolgu turn'lerin ort. φ (entropi): {np.delete(phi_entropy, spike_idx).mean():.3f}")

    # --- Görselleştirme ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.arange(T)
    colors = ["#C44E52" if i in spike_idx else "#4C72B0" for i in range(T)]
    ax.bar(x - 0.2, phi_entropy, width=0.4, label="entropi-Δt", color=colors)
    ax.bar(x + 0.2, phi_uniform, width=0.4, label="uniform-Δt",
           color="#dddddd", edgecolor="gray")
    ax.set_xlabel("Turn"); ax.set_ylabel("bağlanma oranı  φ")
    ax.set_title("Turn başına denge-noktasına bağlanma\n(kırmızı = yüksek-surprisal turn)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(surprisals, phi_entropy, c="#C44E52", label=f"entropi-Δt (r={r_entropy:+.2f})")
    ax.scatter(surprisals, phi_uniform, c="#888888", label=f"uniform-Δt (r={r_uniform:+.2f})")
    ax.set_xlabel("Surprisal  S(u)"); ax.set_ylabel("bağlanma oranı  φ")
    ax.set_title("Surprisal arttıkça bağlanma artıyor mu?")
    ax.legend(); ax.grid(alpha=0.3)

    savefig("sim1_entropy_dt_mapping.png")

    ok = bool((r_entropy > 0.7) and (r_entropy - r_uniform > 0.3))
    verdict(ok,
            "Entropi-Δt, yüksek-surprisal turn'lerin sinyale bağlanmasını belirgin "
            "şekilde artırıyor; uniform-Δt bunu yapamıyor. Haritalama sağlam.")
    return {"r_entropy": r_entropy, "r_uniform": r_uniform, "passed": ok}


if __name__ == "__main__":
    run()
