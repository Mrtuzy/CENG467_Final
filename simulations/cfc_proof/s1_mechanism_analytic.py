"""
S1 — Mekanizma analitik: entropi→Δt bir kaldıraçtır, ama tek başına kanıt değil
=============================================================================
İDDİA (türetilen)
    φ(Δt) = 1 - e^{-λΔt} monoton-artan ve içbükeydir. Δt = Δt_min + β·S
    eşlemesi altında φ, surprisal S'in monoton (doğrusal-olmayan) bir
    fonksiyonudur. Yani "yüksek-surprisal turn'e daha çok bağlan" DOĞRUDAN
    matematikten gelir — bunu ÖLÇMEK bir tautolojidir.

DÜRÜST ÇERÇEVE (v1'in düzeltilmesi)
    v1'de φ ↔ surprisal korelasyonunu (+0.89) "bulgu" olarak sunmak yanıltıcı:
    o korelasyon tasarımdan gelir. Burada onu *türetip* iki gerçek soruyu
    ayırıyoruz:
      (a) MEKANİZMA: Δt kaldıracı φ'yi gerçekten hareket ettiriyor mu? (evet, analitik)
      (b) DOYUM: kaldıraç nerede işe yaramaz hale gelir? → φ→1 doygunluğu.
    Asıl kanıt yükü ("surprisal ↔ ÖNEM") S3'e devredilir; bu sim onu açıkça söyler.

ÇIKTI → TASARIM
    Doyum analizi, surprisal'ın çalışma aralığında φ'nin tavana dayanmaması
    için Δt'nin makul bir bandda kalması gerektiğini gösterir (S2/S5'e girdi).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import (LinearCTNode, dt_linear, DELTA_T_MIN, BETA, LAMBDA,
                  savefig, verdict, C_LIQUID, C_BASE, C_ALT)


def run(seed: int = 0):
    # Gerçekçi surprisal dağılımı (DistilGPT-2 ölçeğine yakın, ağır kuyruklu)
    rng = np.random.default_rng(seed)
    S = np.clip(rng.gamma(2.0, 1.2, size=4000), 0.05, None)

    dt = dt_linear(S, beta=BETA, dt_min=DELTA_T_MIN)
    phi = LinearCTNode.phi_analytic(dt, lam=LAMBDA)

    # Doyum: φ ≥ 0.95 olan turn oranı (bu turn'ler artık birbirinden ayırt edilemez)
    sat_frac = float(np.mean(phi >= 0.95))
    # φ'nin surprisal'a duyarlılığı — ANALİTİK türev:
    #   φ = 1 - e^{-λΔt},  Δt = Δt_min + βS  ⇒  dφ/dS = λβ e^{-λΔt} = λβ(1-φ).
    # (np.gradient yerine kapalı form: tekrarlı S değerlerinde nan üretmez.)
    order = np.argsort(S)
    Ss, phis = S[order], phi[order]
    dphi_dS = LAMBDA * BETA * (1.0 - phis)

    print(f"  Surprisal aralığı: [{S.min():.2f}, {S.max():.2f}], medyan {np.median(S):.2f}")
    print(f"  φ doygunluğu (φ≥0.95) turn oranı: {sat_frac*100:.1f}%")
    print(f"  Ortalama dφ/dS (ayırt-etme gücü): {np.mean(dphi_dS):+.4f}")

    # --- Görselleştirme: iki panel ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (1) φ(Δt) eğrisi + surprisal→Δt eşlemesinin düştüğü bölge
    dgrid = np.linspace(0, dt.max(), 300)
    ax[0].plot(dgrid, LinearCTNode.phi_analytic(dgrid, LAMBDA), color=C_LIQUID, lw=2.5,
               label=r"$\varphi=1-e^{-\lambda\Delta t}$")
    ax[0].axhline(0.95, color=C_BASE, ls="--", lw=1, label=r"saturation threshold $\varphi=0.95$")
    # surprisal kantilleri nereye düşüyor?
    for q in [0.1, 0.5, 0.9, 0.99]:
        dq = dt_linear(np.quantile(S, q))
        ax[0].axvline(dq, color=C_ALT, alpha=0.25)
        ax[0].text(dq, 0.05, f"S{int(q*100)}", rotation=90, va="bottom", fontsize=8, color=C_ALT)
    ax[0].set_xlabel(r"$\Delta t = \Delta t_{min} + \beta S$")
    ax[0].set_ylabel(r"commitment $\varphi$")
    ax[0].set_title("Mechanism is analytic (derivation, not tautology)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # (2) φ'nin surprisal'a duyarlılığı — yüksek S'de tükeniyor
    ax[1].plot(Ss, dphi_dS, color=C_LIQUID, lw=2)
    ax[1].fill_between(Ss, 0, dphi_dS, color=C_LIQUID, alpha=0.15)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("surprisal  S")
    ax[1].set_ylabel(r"$d\varphi/dS$  (discrimination power)")
    ax[1].set_title("Lever saturates at high surprisal → motivates S5")
    ax[1].grid(alpha=0.3)
    savefig("s1_mechanism_analytic.png")

    # Karar: mekanizma var (dφ/dS pozitif başlar) ve doyum gerçek bir risk
    ok = np.mean(dphi_dS[:len(dphi_dS)//4]) > 0 and sat_frac > 0
    verdict(ok, "Δt kaldıracı φ'yi analitik olarak hareket ettiriyor (mekanizma "
                "sağlam) AMA yüksek surprisal'da doyuyor: bu yüzden 'φ↔surprisal "
                "korelasyonu' bir kanıt değil tasarım sonucudur; asıl kanıt "
                "surprisal↔ÖNEM ilişkisidir (S3).")
    return {"sat_frac": sat_frac, "mean_dphi_dS": float(np.mean(dphi_dS)),
            "passed": ok}


if __name__ == "__main__":
    run()
