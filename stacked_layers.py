"""
Stacked-attention experiment: does n_eff stay low across depth, so that
P_succ = n_eff/T compounds toward (n_eff/T)^L, or does the score
distribution flatten as it propagates, washing the exponential out?

We feed each layer's pooled output as the next layer's input and measure
mean n_eff per layer. Two architectural variants:
  - pure pooling:  X <- rownorm(Z)
  - residual:      X <- rownorm(X + Z)         (as in real Transformers)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

def neff_and_weights(S, beta):
    M = S - S.max(axis=1, keepdims=True)
    E = np.exp(beta * M)
    return E.sum(axis=1), E / E.sum(axis=1, keepdims=True)   # per-row n_eff in [1,T]

def rownorm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def run_stack(T, d_k, beta, L, residual, rng, project=False):
    X = rownorm(rng.standard_normal((T, d_k)))
    neff_layer = []
    for _ in range(L):
        if project:
            Wq = rng.standard_normal((d_k, d_k)) / np.sqrt(d_k)
            Wk = rng.standard_normal((d_k, d_k)) / np.sqrt(d_k)
            Wv = rng.standard_normal((d_k, d_k)) / np.sqrt(d_k)
            Q, K, V = X @ Wq, X @ Wk, X @ Wv
        else:
            Q = K = V = X
        S = rownorm(Q) @ rownorm(K).T
        ne, W = neff_and_weights(S, beta)
        neff_layer.append(ne.mean())
        Z = W @ V
        X = rownorm(X + Z) if residual else rownorm(Z)
    return np.array(neff_layer)

def ci(a, conf=0.95):
    a = np.asarray(a); m = a.mean(0)
    h = sem(a, 0) * t.ppf((1+conf)/2., a.shape[0]-1)
    return m, h

T, d_k, L, trials = 32, 4, 12, 200
betas = [2.0, 5.0, 10.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
variants = [("pure pooling", False), ("residual", True)]
colors = {2.0:"tab:blue", 5.0:"tab:green", 10.0:"tab:red"}

for ax, (vname, res) in zip(axes, variants):
    print(f"\n=== variant: {vname} ===")
    for b in betas:
        rng = np.random.default_rng(0)
        runs = np.array([run_stack(T, d_k, b, L, res, rng) for _ in range(trials)])
        m, h = ci(runs)
        ax.errorbar(range(1, L+1), m/T, yerr=h/T, marker="o", capsize=3,
                    color=colors[b], label=f"$\\beta$={b}")
        print(f"  beta={b:4.1f}: n_eff/T  L1={m[0]/T:.3f}  L6={m[5]/T:.3f}  L12={m[-1]/T:.3f}")
        # measured cumulative AA cost vs naive layer-1 extrapolation
        meas_cost = np.prod(np.sqrt(T/m))              # product over layers
        naive_cost = (T/m[0])**(L/2.)
        print(f"           measured cumulative AA factor = {meas_cost:.2e} ; "
              f"naive (T/n_eff_1)^(L/2) = {naive_cost:.2e}")
    ax.axhline(1.0, ls=":", c="gray", lw=1)
    ax.set_xlabel("layer index $\\ell$"); ax.set_ylabel("mean $n_{eff}/T$ (=$P_{succ}$)")
    ax.set_title(f"{vname}  ($T$={T}, $d_k$={d_k})")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()

fig.suptitle("Does attention sharpness survive depth? "
             "Per-layer $P_{succ}=n_{eff}/T$ across a stacked attention network",
             fontsize=13)
fig.tight_layout()
fig.savefig("figD_stacked_depth.png", dpi=200, bbox_inches="tight")
print("\nsaved figD_stacked_depth.png")
