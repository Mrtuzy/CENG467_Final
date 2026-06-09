"""
Simülasyon 2 — β anlamlı/monoton bir kontrol mü, yoksa dejenere mi?
===================================================================
HİPOTEZ
    β (entropi→Δt kazancı) arttıkça, yüksek-entropili turn'ler ile dolgu
    turn'ler arasındaki AYRIM (bağlanma oranı φ farkı) monoton artar; ancak
    çok büyük β'da φ tavanına (1.0) doyduğu için ayrım saturasyona girer.
    Yani anlamlı bir orta rejim vardır (β=1 makul).

NEDEN ÖNEMLİ
    β=0 ise Δt sabittir → entropi hiç kullanılmaz. β çok büyükse her turn
    tam bağlanır → ayrım kaybolur. Tasarım ancak "tatlı nokta" gerçekten
    varsa savunulabilir.

ÇÜRÜTME KOŞULU
    Ayrım β'dan bağımsızsa (düz çizgi) veya monoton azalıyorsa, β anlamsızdır.

ÖLÇÜT
    Ayrım = ⟨φ⟩_spike − ⟨φ⟩_dolgu.  (φ = denge-noktasına bağlanma oranı,
    bkz. sim1 / sim_common.LTCCell.commitment_fraction)
"""
import numpy as np
import matplotlib.pyplot as plt

from simulations.cfc_sims.sim_common import LTCCell, surprisal_to_dt, savefig, verdict, DELTA_T_MIN


def _commitment_profile(cell, inputs, dts):
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
    spike_idx = np.array([2, 7, 10])
    surprisals[spike_idx] = rng.uniform(4.0, 5.5, size=len(spike_idx))
    fill_mask = np.ones(T, bool); fill_mask[spike_idx] = False

    cell = LTCCell(dim=dim, seed=seed)
    betas = np.linspace(0.0, 4.0, 25)
    separations = []
    for b in betas:
        dts = surprisal_to_dt(surprisals, beta=b, dt_min=DELTA_T_MIN)
        phi = _commitment_profile(cell, inputs, dts)
        separations.append(phi[spike_idx].mean() - phi[fill_mask].mean())
    separations = np.array(separations)

    sep0 = separations[0]
    sep_peak = separations.max()
    beta_peak = betas[separations.argmax()]
    sep_at_1 = separations[np.argmin(np.abs(betas - 1.0))]

    print(f"  β=0 (uniform) ayrım   : {sep0:+.4f}")
    print(f"  β=1 (varsayılan) ayrım: {sep_at_1:+.4f}")
    print(f"  tepe ayrım            : {sep_peak:+.4f}  (β≈{beta_peak:.2f})")

    plt.figure(figsize=(8, 4.5))
    plt.plot(betas, separations, marker="o", color="#4C72B0")
    plt.axvline(1.0, ls="--", color="#C44E52", label="β=1 (varsayılan)")
    plt.axhline(sep0, ls=":", color="gray", label="β=0 (uniform) referansı")
    plt.scatter([beta_peak], [sep_peak], color="#55A868", zorder=5,
                label=f"tepe (β≈{beta_peak:.2f})")
    plt.xlabel("β  (entropi→Δt kazancı)")
    plt.ylabel("Ayrım:  ⟨φ⟩_spike − ⟨φ⟩_dolgu")
    plt.title("β, entropi-hassasiyetini monoton kontrol ediyor mu?")
    plt.legend(); plt.grid(alpha=0.3)
    savefig("sim2_beta_sensitivity.png")

    rising = separations[5] > separations[0]
    meaningful = (sep_at_1 - sep0) > 0.4 * (sep_peak - sep0 + 1e-9)
    ok = bool(rising and meaningful and (sep_peak > sep0 + 1e-3))
    verdict(ok,
            f"β=0'da ayrım ~0; β büyüdükçe ayrım artıp ~β={beta_peak:.1f} civarında "
            f"doyuyor. Anlamlı bir orta rejim var, β=1 makul.")
    return {"sep0": float(sep0), "sep_at_1": float(sep_at_1),
            "beta_peak": float(beta_peak), "passed": ok}


if __name__ == "__main__":
    run()
