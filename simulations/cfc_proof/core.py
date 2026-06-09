"""
core.py — CfC kanıt-suite'inin ortak altyapısı (v2)
====================================================
Bu modül üç şey sağlar:

1.  ANALİTİK substrat (``LinearCTNode``): tek turn'ün sabit girdiyle
    işlenmesini *kapalı formda* çözer. Sürekli-zamanlı sızdıran (leaky)
    integratör için

        dx/dt = -λ (x - x*(I))   ⇒   x(Δt) = x* + (x0 - x*) e^{-λΔt}

    olduğundan, bir turn'ün "bağlanma oranı" (commitment) DOĞRUDAN analitiktir:

        φ(Δt) = ‖x(Δt) - x0‖ / ‖x* - x0‖ = 1 - e^{-λΔt}.

    Bu, v1'deki φ'yi *ampirik* ölçmek yerine matematiksel olarak türetmemizi
    sağlar — "tautoloji"yi ölçmek yerine sahiplenir, asıl sorulması gereken
    soruyu (surprisal ↔ önem korelasyonu) açığa çıkarır.

2.  GERÇEK model substratı (``RealCfC``): projenin eğittiği ``ncps`` CfC'sini
    rastgele/sabit ağırlıklarla, eğitimsiz çalıştırır. Böylece mekanizmanın
    LTC'ye özgü bir yapay sonuç değil, gerçek CfC'de de var olan bir özellik
    olduğunu gösterebiliriz ("sims LTC ama model CfC" açığını kapatır).

3.  KAPALI-FORM etki (influence): doğrusal zaman-değişken bir zincirde her
    turn'ün son duruma katkısı analitik olarak hesaplanır — eğitilmiş bir
    okuyucu (readout) olmadan, önem-kurtarma (recovery) ölçmek için.

Ayrıca düzgün istatistik (bootstrap GA, Cohen's d) ve figür yardımcıları.
Bağımlılık: numpy, scipy, matplotlib. ``RealCfC`` ek olarak torch + ncps ister.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as _st

# --- src/config.py ile senkron tutulan sabitler (modelsiz kalmak için kopya) ---
DELTA_T_MIN = 0.1
BETA = 1.0
TAU = 0.5
LAMBDA = 1.0  # sızdıran integratör çürüme oranı (substrat sabiti)

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Tutarlı, sunuma uygun renkler
C_LIQUID = "#C44E52"   # entropi-Δt (liquid)
C_BASE = "#888888"     # uniform / baseline
C_ALT = "#4C72B0"      # ikincil
C_OK = "#55A868"       # pozitif/kabul


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


# ===================================================================== #
#  Δt eşlemeleri (mapping fonksiyonları) — S5 bunları karşılaştırır       #
# ===================================================================== #
def dt_linear(S, beta=BETA, dt_min=DELTA_T_MIN):
    """Projenin mevcut eşlemesi: Δt = Δt_min + β·S (sınırsız, doğrusal)."""
    return dt_min + beta * np.asarray(S, float)


def dt_bounded(S, beta=BETA, dt_min=DELTA_T_MIN, scale=None):
    """Sınırlı eşleme: Δt = Δt_min + β·s0·tanh(S/s0). Aykırı surprisal'ı doyurur."""
    S = np.asarray(S, float)
    s0 = scale if scale is not None else (np.median(S) + 1e-6)
    return dt_min + beta * s0 * np.tanh(S / s0)


def dt_rank(S, beta=BETA, dt_min=DELTA_T_MIN):
    """Sıra-normalize eşleme: Δt, S'in [0,1] yüzdelik sırasına göre. Ölçek-bağımsız."""
    S = np.asarray(S, float)
    if S.size <= 1:
        return dt_min + beta * 0.5 * np.ones_like(S)
    ranks = _st.rankdata(S, method="average") / S.size  # (0,1]
    return dt_min + beta * ranks


MAPPINGS = {"linear": dt_linear, "bounded": dt_bounded, "rank": dt_rank}


# ===================================================================== #
#  Substrat 1 — Analitik doğrusal sürekli-zaman düğümü                    #
# ===================================================================== #
class LinearCTNode:
    """Sızdıran integratör: dx/dt = -λ(x - x*(I)),  x*(I) = A ⊙ tanh(W I).

    Tek turn'ün analitik çözümü kapalı formda bilindiği için bağlanma oranı
    φ(Δt) = 1 - e^{-λΔt} *kanıtlanabilir*. Ağırlıklar sabit (öğrenilmemiş);
    amaç dinamiğin yapısını izole etmek.
    """

    def __init__(self, dim=8, lam=LAMBDA, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.lam = lam
        self.W = rng.normal(0, 1.0, size=(dim, dim))
        self.A = rng.uniform(0.5, 1.5, size=dim)

    def target(self, I):
        """Girdi I için çekim noktası x*(I)."""
        return self.A * np.tanh(self.W @ I)

    def step(self, x0, I, dt):
        """x0'dan başlayıp I sabitken Δt boyunca KAPALI-FORM ilerlet."""
        xstar = self.target(I)
        return xstar + (x0 - xstar) * np.exp(-self.lam * dt)

    @staticmethod
    def phi_analytic(dt, lam=LAMBDA):
        """Bağlanma oranı (kapalı form): φ = 1 - e^{-λΔt}."""
        return 1.0 - np.exp(-lam * np.asarray(dt, float))

    def commitment_fraction(self, x0, I, dt):
        """φ = ‖x(Δt)-x0‖ / ‖x*-x0‖. Analitik değerle bire bir örtüşür."""
        xstar = self.target(I)
        x = self.step(x0, I, dt)
        denom = np.linalg.norm(xstar - x0) + 1e-9
        return float(np.linalg.norm(x - x0) / denom)


# ===================================================================== #
#  Substrat 2 — Gerçek ncps CfC (eğitimsiz, sabit ağırlık)               #
# ===================================================================== #
class RealCfC:
    """Projenin eğittiği ``ncps`` CfC'si — rastgele sabit ağırlıkla, eval modda.

    Mekanizmanın (Δt → bağlanma) gerçek modelde de var olduğunu doğrular.
    torch + ncps gerektirir; yoksa ``available()`` False döner.
    """

    def __init__(self, input_size=8, units=32, seed=0):
        import torch
        from ncps.torch import CfC

        torch.manual_seed(seed)
        self.torch = torch
        self.model = CfC(input_size, units, batch_first=True)
        self.model.eval()
        self.input_size = input_size
        self.units = units

    @staticmethod
    def available():
        try:
            import torch  # noqa
            from ncps.torch import CfC  # noqa
            return True
        except Exception:
            return False

    def run_sequence(self, inputs, dts):
        """inputs (T,d), dts (T,) → states (T+1,units), per-turn ‖Δh‖ (T,)."""
        torch = self.torch
        x = torch.tensor(np.asarray(inputs, np.float32))[None]      # (1,T,d)
        ts = torch.tensor(np.asarray(dts, np.float32))[None]        # (1,T)
        with torch.no_grad():
            out, _ = self.model(x, timespans=ts)                    # (1,T,units)
        seq = out[0].cpu().numpy()
        states = np.vstack([np.zeros(self.units), seq])
        disp = np.linalg.norm(np.diff(states, axis=0), axis=1)
        return states, disp

    def single_turn_phi(self, I, dt, dt_inf=60.0):
        """Tek turn için bağlanma oranı: ‖h(Δt)‖ / ‖h(∞)‖ (x0=0'dan)."""
        torch = self.torch
        x = torch.tensor(np.asarray(I, np.float32))[None, None]     # (1,1,d)
        with torch.no_grad():
            h_dt, _ = self.model(x, timespans=torch.tensor([[float(dt)]]))
            h_inf, _ = self.model(x, timespans=torch.tensor([[float(dt_inf)]]))
        n_inf = float(np.linalg.norm(h_inf[0, 0].numpy())) + 1e-9
        return float(np.linalg.norm(h_dt[0, 0].numpy()) / n_inf)


# ===================================================================== #
#  Kapalı-form etki (influence) — eğitimsiz önem-kurtarma için           #
# ===================================================================== #
READOUT_NOISE = 0.06  # turn-başına okuyucudaki temsil/gate gürültü tabanı


def readout_signal(dts, content_norms=None, lam=LAMBDA, noise=0.0, rng=None,
                   budget=None):
    """Turn-BAŞINA okunabilir önem sinyali (gerçek CfCPruner'a sadık abstraksiyon).

    CfCPruner her turn için bir skor üretir; turn i'nin kendi durumuna bağlanması
    o turn'ün Δt_i'si boyunca ne kadar yol aldığıyla, yani KAPALI formda
        commit_i = 1 - e^{-λ Δt_i}
    ile belirlenir (son-duruma katkı değil — okuma turn anında yapılır, bu yüzden
    recency çürümesi yoktur). Bu, son-durum ``chain_influence``'ından farklı ve
    pruning görevi için DOĞRU olandır.

    Parametreler
    ------------
    content_norms : verilirse sinyal içerik büyüklüğüyle ölçeklenir (input kanalı).
                    Verilmezse eşit → önem yalnızca Δt kanalından akar.
    noise         : okuyucu gürültü tabanı (σ). β çok küçükken sinyali bastırır.
    budget        : verilirse Δt'ler toplamı sabit tutulur (sabit hesap bütçesi);
                    eşlemenin ÖLÇEĞİNİ değil ŞEKLİNİ izole eder (S5 kullanır).
    """
    dts = np.asarray(dts, float)
    if budget is not None and dts.sum() > 0:
        dts = dts * (budget / dts.sum())
    commit = 1.0 - np.exp(-lam * dts)
    sig = commit if content_norms is None else commit * np.asarray(content_norms, float)
    if noise > 0:
        r = rng if rng is not None else np.random.default_rng()
        sig = sig + r.normal(0, noise, size=sig.shape)
    return sig


def chain_influence(dts, content_norms=None, lam=LAMBDA):
    """Doğrusal sızdıran-integratör zincirinde her turn'ün SON duruma katkısı.

    Turn i sabit hedef a*_i'ye Δt_i boyunca yaklaşır:
        x_i = x_{i-1} e^{-λΔt_i} + a*_i (1 - e^{-λΔt_i}).
    Dolayısıyla turn i'nin son duruma (x_T) katkı katsayısı KAPALI formda:
        coeff_i = (1 - e^{-λΔt_i}) · Π_{k>i} e^{-λΔt_k}
                = (bağlanma_i) × (sonraki turn'lerle çürüme / recency).
    influence_i = coeff_i · ‖a*_i‖.

    Bu, eğitilmiş bir okuyucu olmadan "model durumu hangi turn'lere ne kadar
    bağlandı" sorusunun *analitik* cevabıdır. content_norms verilmezse içerik
    büyüklüğü eşit alınır → önem yalnızca Δt kanalından (yani surprisal'dan)
    akabilir; böylece Δt kaldıracını izole ederiz.
    """
    dts = np.asarray(dts, float)
    T = dts.size
    commit = 1.0 - np.exp(-lam * dts)                       # (T,)
    decay_after = np.exp(-lam * dts)                        # her turn'ün çürütücüsü
    # Π_{k>i} decay_k = suffix product
    suffix = np.ones(T)
    acc = 1.0
    for i in range(T - 1, -1, -1):
        suffix[i] = acc
        acc *= decay_after[i]
    coeff = commit * suffix
    if content_norms is not None:
        coeff = coeff * np.asarray(content_norms, float)
    return coeff


# ===================================================================== #
#  Sentetik üretici süreçler                                            #
# ===================================================================== #
def make_surprisal(u, rho, rng):
    """Latent önem u ile hedeflenen ~rho Spearman ilişkili pozitif surprisal üret.

    Gauss kopulası: z_S = rho·z_u + sqrt(1-rho²)·z_eps (her ikisi standardize),
    sonra pozitifliğe taşı. rho=1 → mükemmel proxy, rho=0 → bağımsız gürültü.
    """
    u = np.asarray(u, float)
    zu = (_st.rankdata(u) - 0.5) / u.size
    zu = _st.norm.ppf(np.clip(zu, 1e-4, 1 - 1e-4))
    zu = (zu - zu.mean()) / (zu.std() + 1e-9)
    eps = rng.standard_normal(u.size)
    z = rho * zu + np.sqrt(max(0.0, 1 - rho ** 2)) * eps
    return softplus(1.2 * z + 0.5)  # pozitif surprisal benzeri


def heavy_tailed_importance(T, rng, n_strong=None):
    """Ağır kuyruklu önem: çoğu turn küçük, birkaçı güçlü (gerçek diyaloglar gibi)."""
    a = rng.exponential(0.4, size=T)
    k = n_strong if n_strong is not None else max(1, T // 6)
    idx = rng.choice(T, size=k, replace=False)
    a[idx] += rng.uniform(2.0, 4.0, size=k)
    return np.clip(a, 1e-3, None)


# ===================================================================== #
#  İstatistik yardımcıları                                              #
# ===================================================================== #
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(_st.spearmanr(a, b).statistic)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(_st.pearsonr(a, b).statistic)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Ortalama için bootstrap %95 güven aralığı. → (mean, lo, hi)."""
    v = np.asarray(values, float)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def cohens_d(x, y):
    """Standartlaştırılmış etki büyüklüğü (gruplar arası ayrım)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = x.size, y.size
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2) + 1e-12)
    return float((x.mean() - y.mean()) / (sp + 1e-12))


def paired_p(a, b):
    """Eşli Wilcoxon işaret-sıra testi p-değeri (a > b kalıcı mı?)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        return float(_st.wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


# ===================================================================== #
#  Figür / karar yardımcıları                                          #
# ===================================================================== #
def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 figür → figures/{name}")
    return path


def verdict(passed, msg):
    tag = "✓ TUTARLI" if passed else "✗ TUTARSIZ"
    print(f"\n[{tag}] {msg}\n")
    return bool(passed)
