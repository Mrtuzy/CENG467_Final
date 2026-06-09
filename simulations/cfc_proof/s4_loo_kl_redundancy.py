"""
S4 — Leave-one-out KL'in gerçekten ölçtüğü şey (redundans testi)
===============================================================
v1'İN ZAYIFLIĞI
    v1, p=softmax(Σ a_i v_i) üzerinde KL'in latent önem a'yı +0.91 kurtardığını
    gösterdi — ama bu büyük ölçüde tautoloji: bağımsız turn'lerde KL_j zaten a_j
    ile birlikte büyür. Gerçek ve sinsi konfaunt REDUNDANS'tır.

YENİ SORU
    İki turn AYNI bilgiyi taşıyorsa (redundant), birini tek başına çıkarmanın KL'i
    DÜŞÜK olur — diğeri bilgiyi taşımaya devam eder — turn "önemli" olsa bile.
    Bu, leave-one-out'un bilinen kör noktasıdır. Öyleyse teacher etiketimiz NEYİ
    ölçüyor: latent önem a'yı mı, yoksa "bu turn cevap için MARJİNAL olarak gerekli
    mi"yi mi?

YÖNTEM (modelsiz)
    K-sınıflı cevap p=softmax(Σ a_i v_i). Turn'lerin bir kısmı yön (v) paylaşır
    → redundans. Redundans seviyesini süpür. Ölç:
      • Spearman(a, KL):  latent önemi kurtarma (redundansla DÜŞMELİ — dürüst sınır)
      • Spearman(TV, KL): KL'in gerçek cevap-etkisini (total-variation) kurtarması
        (redundanstan BAĞIMSIZ yüksek kalmalı)
    Ayrıca pipeline'ın [0,1] min-max normalizasyonunun sırayı bozmadığını doğrula.

ÇIKTI → TASARIM
    Teacher hedefi "ham önem" değil "marjinal cevap-gerekliliği"dir. Bu bir HATA
    değil, pruning için DOĞRU hedeftir: redundant turn düşük hedef alır ve onu
    budamak GÜVENLİDİR (bilgi diğer turn'de hayatta kalır). Bu nüans, "KL=önem"
    iddiasından bilimsel olarak daha sağlamdır.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from core import spearman, bootstrap_ci, savefig, verdict, C_LIQUID, C_BASE, C_ALT


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _one_context(rng, redund_frac, theta=1.5):
    """Fact-coverage + fact-başına DOYUM cevap modeli.

    Cevap, gizli 'fact'ler üzerindeki bir dağılımdır: p = softmax(g(evidence_f)),
    g(e) = θ·tanh(e/θ) fact-başına DOYAN bir kanıt fonksiyonudur. evidence_f, o
    fact'i taşıyan turn'lerin önem toplamıdır. Her turn BİR fact taşır; redundans ↑
    → farklı fact sayısı ↓ (turn'ler aynı fact'i paylaşır).

    Doyum kritik: bir fact yeterli kanıt topladıysa (e≳θ), fazladan bir taşıyıcıyı
    ÇIKARMAK cevabı neredeyse değiştirmez → o redundant turn'ün KL'i düşük olur,
    önemi (a) yüksek olsa bile. Bu, leave-one-out'un gerçek kör noktasıdır.
    """
    T = rng.integers(9, 16)
    a = np.clip(rng.exponential(0.5, size=T) + 0.05, 0.05, None)
    a[rng.integers(0, T)] += rng.uniform(2, 4)  # en az bir güçlü turn

    # Fact ataması: redundans ↑ → distinct fact ↓ (turn'ler fact paylaşır)
    n_facts = max(2, int(round((1.0 - redund_frac) * T)))
    fact_of = rng.integers(0, n_facts, size=T)

    def evidence(active):
        ev = np.zeros(n_facts)
        for i in range(T):
            if active[i]:
                ev[fact_of[i]] += a[i]
        return theta * np.tanh(ev / theta)  # fact-başına doyum

    full = np.ones(T, bool)
    p_full = _softmax(evidence(full))

    kl, tv = np.zeros(T), np.zeros(T)
    for j in range(T):
        act = full.copy(); act[j] = False
        p_mj = _softmax(evidence(act))
        kl[j] = float(np.sum(p_full * (np.log(p_full + 1e-12) - np.log(p_mj + 1e-12))))
        tv[j] = 0.5 * float(np.abs(p_full - p_mj).sum())

    return spearman(a, kl), spearman(tv, kl), a, kl


def run(seed: int = 0, n_seeds: int = 6):
    redunds = np.linspace(0.0, 0.8, 9)
    rec_a, rec_tv = [], []
    a_lo, a_hi = [], []
    for r in redunds:
        sa, stv = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s)
            for _ in range(150):
                ca, ctv, *_ = _one_context(rng, r)
                sa.append(ca); stv.append(ctv)
        m, lo, hi = bootstrap_ci(sa)
        rec_a.append(m); a_lo.append(lo); a_hi.append(hi)
        rec_tv.append(np.mean(stv))
        print(f"  redundans={r:.1f} | Spearman(a,KL)={m:+.3f}  "
              f"Spearman(TV,KL)={np.mean(stv):+.3f}")
    rec_a, rec_tv = np.array(rec_a), np.array(rec_tv)

    # Normalizasyon sıra-koruma kontrolü (pipeline [0,1] min-max yapıyor)
    rng = np.random.default_rng(seed)
    _, _, a_demo, kl_demo = _one_context(rng, 0.3)
    kl_norm = (kl_demo - kl_demo.min()) / (np.ptp(kl_demo) + 1e-12)
    norm_rho = spearman(kl_demo, kl_norm)
    print(f"\n  [0,1] min-max normalizasyon sıra-koruması: ρ={norm_rho:+.3f} (≈+1 beklenir)")

    # --- Görselleştirme ---
    plt.figure(figsize=(8.5, 5))
    plt.plot(redunds, rec_a, marker="o", color=C_LIQUID, lw=2.4,
             label="Spearman(latent importance a, KL)")
    plt.fill_between(redunds, a_lo, a_hi, color=C_LIQUID, alpha=0.18)
    plt.plot(redunds, rec_tv, marker="^", color=C_ALT, lw=2.4,
             label="Spearman(answer-impact TV, KL)")
    plt.axhline(0, color="k", lw=0.6)
    plt.xlabel("redundancy fraction (share of turns carrying the same info)")
    plt.ylabel("rank correlation")
    plt.title("S4 — KL measures 'marginal answer-necessity', not 'raw importance'")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.ylim(-0.05, 1.02)
    savefig("s4_loo_kl_redundancy.png")

    ok = (rec_tv.min() > 0.9 and rec_a[0] > 0.75
          and rec_a[-1] < rec_a[0] - 0.1 and norm_rho > 0.999)
    verdict(ok, "KL, redundanstan bağımsız olarak gerçek cevap-etkisini (TV) "
                f"sadakatle kurtarıyor (≥{rec_tv.min():.2f}); latent önemi kurtarması "
                f"ise redundansla düşüyor ({rec_a[0]:.2f}→{rec_a[-1]:.2f}). Yani teacher "
                "hedefi = marjinal cevap-gerekliliği — pruning için DOĞRU hedef. "
                "Min-max normalizasyon sırayı bozmuyor.")
    return {"redunds": redunds.tolist(), "recover_a": rec_a.tolist(),
            "recover_tv": rec_tv.tolist(), "norm_rho": float(norm_rho), "passed": ok}


if __name__ == "__main__":
    run()
