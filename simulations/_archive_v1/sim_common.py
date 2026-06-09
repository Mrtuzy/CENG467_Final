"""
Ortak simülasyon altyapısı
==========================
Bu modül, projenin merkezindeki Liquid Time-Constant (LTC) ODE'sini
*öğrenilmiş ağırlık olmadan*, sabit ve makul parametrelerle uygular.
Amaç bir modeli eğitmek değil; tasarladığımız mekanizmanın
(entropi → Δt → sürekli-zamanlı işleme) kavramsal olarak sağlam olup
olmadığını izole etmek.

LTC dinamiği (Hasani et al., 2021):

    dx/dt = -x/τ + S(x, I) ⊙ (A - x)

burada S(x, I) = σ(γ · (W_x x + W_I I + b))  ∈ (0, 1) girdiye-bağlı
"liquid" kapısıdır. Efektif zaman sabiti τ_sys = 1 / (1/τ + S) olduğundan
güçlü girdi → hızlı tepki ("liquid time-constant" özelliği).

Her turn, kendi Δt_i süresi boyunca girdisi sabit tutularak entegre edilir.
Δt_i = Δt_min + β · S_i  (S_i: o turn'ün surprisal'ı).  Bu, src/config.py
ile aynı haritalama.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- src/config.py ile senkron tutulan sabitler (modelsiz kalmak için kopyalandı) ---
DELTA_T_MIN = 0.1
BETA = 1.0
TAU = 0.5  # pruning eşiği (sim4'te referans için)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LTCCell:
    """Sabit (öğrenilmemiş) ağırlıklı tek-katman LTC ODE hücresi.

    Ağırlıklar sabit bir tohumdan üretilir; amaç dinamiğin niteliksel
    davranışını incelemek, performans değil.
    """

    def __init__(self, dim: int = 8, tau: float = 1.0, gamma: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.tau = tau
        self.gamma = gamma
        # Sabit kapı ağırlıkları
        self.W_x = rng.normal(0, 0.5, size=(dim, dim))
        self.W_I = rng.normal(0, 1.0, size=(dim, dim))
        self.b = rng.normal(0, 0.1, size=dim)
        # Sabit ters-potansiyeller (reversal potentials)
        self.A = rng.uniform(-1.0, 1.0, size=dim)

    def _dxdt(self, x, I):
        S = sigmoid(self.gamma * (self.W_x @ x + self.W_I @ I + self.b))
        return -x / self.tau + S * (self.A - x)

    def integrate(self, x, I, dt: float, substeps: int = 60):
        """x durumunu, girdi I sabitken dt süresi boyunca ileri-Euler ile entegre et."""
        h = dt / substeps
        for _ in range(substeps):
            x = x + h * self._dxdt(x, I)
        return x

    def equilibrium(self, I, substeps: int = 600):
        """Girdi I sabitken durumun yakınsadığı denge noktası x* (uzun entegrasyon)."""
        return self.integrate(np.zeros(self.dim), I, dt=60.0, substeps=substeps)

    def commitment_fraction(self, x_before, I, dt, substeps: int = 60, eps: float = 1e-6):
        """Bir turn'ün Δt'si boyunca durumun, o turn'ün denge noktasına ne kadar
        yaklaştığı  φ = ‖x_sonra − x_önce‖ / ‖x* − x_önce‖  ∈ [0, 1].

        Δt → ∞ iken φ → 1 (tam bağlanma), Δt → 0 iken φ → 0. Bu büyüklük, denge
        uzaklığına normalize ettiği için saf Δt (dolayısıyla surprisal) etkisini
        ham ‖Δx‖'ten daha temiz izole eder.
        """
        xeq = self.equilibrium(I)
        x_after = self.integrate(x_before, I, dt, substeps)
        denom = np.linalg.norm(xeq - x_before) + eps
        return float(np.clip(np.linalg.norm(x_after - x_before) / denom, 0.0, 1.0)), x_after

    def run_sequence(self, inputs, dts, x0=None, substeps: int = 60):
        """Bir turn dizisini işle.

        inputs : (T, dim)  her turn'ün girdi vektörü
        dts    : (T,)      her turn'ün Δt'si
        Dönüş  : states (T+1, dim) yörünge,  step_disp (T,) turn başına ‖Δx‖
        """
        T = len(inputs)
        x = np.zeros(self.dim) if x0 is None else np.array(x0, dtype=float)
        states = [x.copy()]
        disp = []
        for i in range(T):
            x_prev = x.copy()
            x = self.integrate(x, inputs[i], dts[i], substeps)
            disp.append(np.linalg.norm(x - x_prev))
            states.append(x.copy())
        return np.array(states), np.array(disp)


def surprisal_to_dt(surprisals, beta: float = BETA, dt_min: float = DELTA_T_MIN):
    """Δt_i = Δt_min + β · S_i  (projenin haritalaması)."""
    return dt_min + beta * np.asarray(surprisals, dtype=float)


def spearman(a, b):
    """Basit Spearman sıra-korelasyonu (scipy'siz)."""
    a, b = np.asarray(a), np.asarray(b)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 figür → {path}")
    return path


def verdict(passed: bool, msg: str):
    tag = "✓ TUTARLI" if passed else "✗ TUTARSIZ"
    print(f"\n[{tag}] {msg}\n")
    return passed
