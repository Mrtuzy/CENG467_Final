"""
Simülasyon 4 — Leave-one-out KL, gerçek önem sıralamasını kurtarıyor mu?
========================================================================
HİPOTEZ
    Teacher etiketleme önermesi: bir turn u_j'yi bağlamdan çıkarınca modelin
    çıktı dağılımı NE KADAR çok değişiyorsa (KL-divergence yüksekse), u_j o
    kadar önemlidir. Bu doğruysa, KL(j) gerçek (latent) önem ağırlığı a_j ile
    monoton artmalı.

NEDEN ÖNEMLİ
    Tüm gold etiketler (teacher_targets.pt) bu varsayıma dayanıyor. Eğer
    leave-one-out KL, önemli ve önemsiz turn'leri ayırt edemiyorsa, CfC'yi
    eğittiğimiz hedef sinyal gürültüdür — pipeline temelden çürür.

ÇÜRÜTME KOŞULU
    KL(j) ile a_j arasında Spearman ≈ 0 ise, LOO-KL önem sinyali taşımıyordur.
    Ayrıca [0,1] min-max normalizasyonunun (kodda yapılan) sıralamayı bozup
    bozmadığını da kontrol ederiz.

YÖNTEM (LLM YOK — sentetik üretici süreç)
    Cevap dağılımını şöyle modelleriz:
        p = softmax( Σ_i a_i · v_i )
    burada v_i ∈ R^V her turn'ün sabit "katkı logit'i", a_i ≥ 0 latent önemi.
    Leave-one-out:  p_{-j} = softmax( Σ_{i≠j} a_i v_i ),  KL(p ‖ p_{-j}).
    KL(j)'yi a_j'ye karşı çiziyoruz.
"""
import numpy as np
import matplotlib.pyplot as plt

from sim_common import spearman, savefig, verdict


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def _one_context(rng, V=50):
    T = rng.integers(6, 14)
    a = rng.exponential(0.5, size=T)            # latent importance (heavy-tailed)
    a[rng.integers(0, T)] += rng.uniform(2, 4)  # en az bir önemli turn
    V_logits = rng.normal(0, 1, size=(T, V))    # her turn'ün katkı logit'i

    full = softmax((a[:, None] * V_logits).sum(axis=0))
    kls = []
    for j in range(T):
        keep = np.ones(T, bool); keep[j] = False
        abl = softmax((a[keep, None] * V_logits[keep]).sum(axis=0))
        kls.append(kl(full, abl))
    return a, np.array(kls)


def run(seed: int = 0, n_contexts: int = 300):
    rng = np.random.default_rng(seed)

    # 1) Toplu Spearman: KL(j) ↔ gerçek importance a_j
    rhos = []
    pooled_a, pooled_kl = [], []
    for _ in range(n_contexts):
        a, kls = _one_context(rng)
        rhos.append(spearman(a, kls))
        pooled_a.extend(a); pooled_kl.extend(kls)
    rhos = np.array(rhos)
    pooled_a = np.array(pooled_a); pooled_kl = np.array(pooled_kl)

    print(f"  Bağlam-içi Spearman( a_j , KL_j ): "
          f"ort={rhos.mean():+.3f}  medyan={np.median(rhos):+.3f}  "
          f"%pozitif={100*(rhos>0).mean():.0f}%")

    # 2) Min-max normalizasyon sıralamayı koruyor mu? (kod [0,1]'e çekiyor)
    a_ex, kl_ex = _one_context(np.random.default_rng(123))
    mn, mx = kl_ex.min(), kl_ex.max()
    normed = (kl_ex - mn) / (mx - mn) if mx > mn else np.full_like(kl_ex, 0.5)
    rho_norm = spearman(kl_ex, normed)
    print(f"  Min-max sonrası sıra korelasyonu (bozulmamalı): {rho_norm:+.3f}")

    # --- Görselleştirme ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(pooled_a, pooled_kl, s=8, alpha=0.3, color="#4C72B0")
    ax.set_xlabel("Gerçek (latent) importance  a_j")
    ax.set_ylabel("Leave-one-out KL_j")
    ax.set_title(f"LOO-KL önemle artıyor mu?\n(toplu Spearman ort.={rhos.mean():+.2f})")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(rhos, bins=25, color="#55A868", edgecolor="white")
    ax.axvline(rhos.mean(), color="#C44E52", ls="--", label=f"ort.={rhos.mean():+.2f}")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("Bağlam-içi Spearman( a , KL )")
    ax.set_ylabel("Bağlam sayısı")
    ax.set_title("Önem sinyali tutarlı mı?")
    ax.legend(); ax.grid(alpha=0.3)
    savefig("sim4_loo_kl_premise.png")

    # --- Karar ---
    ok = rhos.mean() > 0.5 and (rhos > 0).mean() > 0.9 and rho_norm > 0.99
    verdict(ok,
            "Leave-one-out KL, bağlamların büyük çoğunluğunda latent önemi "
            "güçlü ve pozitif sıralıyor; min-max normalizasyon sıralamayı bozmuyor. "
            "Teacher-etiketleme önermesi sağlam.")
    return {"mean_rho": float(rhos.mean()),
            "frac_positive": float((rhos > 0).mean()),
            "rho_norm": rho_norm, "passed": ok}


if __name__ == "__main__":
    run()
