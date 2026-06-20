"""
Simulations for:
"The Cost of Coherence: A No-Go Theorem for Digital Quantum Self-Attention"

Three independent components, each mapped to a paper section.

  Part A  (Sec 3, No-Go theorem):
      Numerical illustration that THREE operations all loosely called
      "applying exp" are mutually distinct:
        (1) element-wise exp:  [exp(S)]_ij
        (2) matrix exp (eigenvalue calculus):  expm(S)
        (3) QSVT (singular-value calculus):    U exp(Sigma) V^dagger
      QSVT implements (3). Softmax needs (1). The figure shows they
      never coincide for T>=2, which is the content of the theorem.

  Part B  (Sec 2, the resource cost -- the previously-missing core):
      Post-selection success probability of the canonical coherent
      construction, P_succ = n_eff / T, where
        n_eff = sum_j exp(beta (s_j - s_max))  in [1, T]
      is the effective number of attended tokens. From it:
        OAA rounds  ~  sqrt(1 / P_succ)            (per layer)
        depth cost  ~  P_succ^{-L/2}               (L stacked layers)
      CAVEAT (must stay in the paper): P_succ = n_eff/T is the success
      probability of THIS construction. It is an UPPER bound on success
      (lower bound on cost) for the canonical block-encode+post-select
      route, NOT yet a proven optimum over all coherent protocols. The
      lower-bound-over-all-protocols claim is an analytic obligation,
      not established by this simulation.

  Part C  (Sec 2 / resource, relabeled from the prior drafts):
      Cost of the COHERENT polynomial (QSP-on-amplitude) route, i.e. the
      only coherent way to get an element-wise exponential. Shows the
      degree must scale ~ linearly with beta (Peak Flattening otherwise).
      Fixes vs prior drafts:
        - no unused pennylane import (this is honest classical NumPy)
        - Born-rule readout: amplitude approximates exp((beta/2) x), the
          weight is |P|^2 / sum|P|^2 -> non-negative by construction,
          replacing the old |P|^1 abs() heuristic
        - degree-vs-beta sweep makes the d ~ O(beta) claim directly,
          instead of a single d=3 vs d=7 snapshot
"""

import numpy as np
from scipy.linalg import expm
from scipy.special import iv
from scipy.stats import sem, t
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)


# ======================================================================
# Shared helpers
# ======================================================================
def gram_scores(T, d_k, rng):
    """Realistic attention score matrix S = Q K^T (generally non-symmetric)."""
    Q = rng.standard_normal((T, d_k)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    K = rng.standard_normal((T, d_k)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    return Q @ K.T


def chebyshev_exp(x, degree, beta):
    """Degree-`degree` Chebyshev (Jacobi-Anger) truncation of exp(beta * x)."""
    approx = iv(0, beta)
    T_prev, T_curr = np.ones_like(np.asarray(x, dtype=float)), np.asarray(x, dtype=float)
    for k in range(1, degree + 1):
        approx = approx + 2 * iv(k, beta) * T_curr
        T_prev, T_curr = T_curr, 2 * x * T_curr - T_prev
    return approx


def ci(samples, confidence=0.95):
    samples = np.asarray(samples)
    m = samples.mean()
    h = sem(samples) * t.ppf((1 + confidence) / 2., len(samples) - 1)
    return m, h


# ======================================================================
# PART A  --  No-Go illustration: three distinct "exp" operations
# ======================================================================
def svt(S, func):
    """Singular-value transform: this is what QSVT actually implements."""
    U, s, Vh = np.linalg.svd(S)
    return (U * func(s)) @ Vh


def part_A():
    print("=" * 70)
    print("PART A : element-wise exp  vs  matrix exp  vs  QSVT (SVT)")
    print("=" * 70)

    def report(S, label):
        # normalise so max singular value = 1 (QSVT domain constraint)
        S = S / np.linalg.norm(S, 2)
        E = np.exp(S)            # (1) element-wise
        M = expm(S)             # (2) matrix exponential (eigenvalue calculus)
        Q = svt(S, np.exp)      # (3) QSVT singular-value transform
        dEM = np.linalg.norm(E - M)
        dEQ = np.linalg.norm(E - Q)
        dMQ = np.linalg.norm(M - Q)
        print(f"\n[{label}]  ||.||_F pairwise distances")
        print(f"   element-wise vs matrix-exp : {dEM:.4f}")
        print(f"   element-wise vs QSVT(SVT)  : {dEQ:.4f}")
        print(f"   matrix-exp   vs QSVT(SVT)  : {dMQ:.4f}")
        return S, E, M, Q

    # (i) a general non-symmetric Gram matrix
    S_gram = gram_scores(6, 4, rng)
    report(S_gram, "general non-symmetric Gram S")

    # (ii) a diagonal matrix -- tests the skeleton's "agree on diagonal" claim
    S_diag = np.diag(rng.standard_normal(4))
    report(S_diag, "diagonal S")

    # (iii) the 2x2 exchange (Pauli-X) used in the proof, for the record
    S_x = np.array([[0., 1.], [1., 0.]])
    Sx, Ex, Mx, Qx = report(S_x, "2x2 exchange matrix (Pauli-X)")
    print("\n   For S=X (after norm, X already has unit singular values):")
    print("   element-wise exp(X) =\n", np.round(Ex, 3))
    print("   matrix exp expm(X)  =\n", np.round(Mx, 3))
    print("   QSVT  U exp(Sig) V^ =\n", np.round(Qx, 3))

    # figure: heatmaps of the three operations on the general Gram matrix
    S = S_gram / np.linalg.norm(S_gram, 2)
    mats = [np.exp(S), expm(S), svt(S, np.exp)]
    titles = ["element-wise  exp(S)$_{ij}$\n(softmax needs THIS)",
              "matrix exp  expm(S)\n(eigenvalue calculus)",
              "QSVT  $U\\,e^{\\Sigma}V^\\dagger$\n(what QSVT DOES)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(m.max() for m in mats)
    vmin = min(m.min() for m in mats)
    for ax, m, ti in zip(axes, mats, titles):
        im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(ti, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("Three distinct operations on one score matrix $S$ "
                 "(general non-symmetric Gram)", fontsize=13)
    fig.savefig("figA_nogo_three_operations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n   -> saved figA_nogo_three_operations.png")


# ======================================================================
# PART B  --  Resource cost: P_succ = n_eff / T, OAA rounds, depth
# ======================================================================
def n_eff(scores, beta):
    return np.exp(beta * (scores - scores.max())).sum()   # in [1, T]


def part_B():
    print("\n" + "=" * 70)
    print("PART B : post-selection success P_succ = n_eff/T, OAA, depth")
    print("=" * 70)

    betas = [1.0, 2.0, 5.0, 10.0]
    Ts = [4, 8, 16, 32, 64, 128]
    d_k = 4
    trials = 300

    psucc = {b: {"m": [], "h": []} for b in betas}
    neff = {b: {"m": [], "h": []} for b in betas}

    for b in betas:
        for T in Ts:
            ne_samples, ps_samples = [], []
            for _ in range(trials):
                # one query against T keys -> length-T score vector
                S = gram_scores(T, d_k, rng)
                scores = S[rng.integers(T)]
                ne = n_eff(scores, b)
                ne_samples.append(ne)
                ps_samples.append(ne / T)
            m_ne, h_ne = ci(ne_samples)
            m_ps, h_ps = ci(ps_samples)
            neff[b]["m"].append(m_ne); neff[b]["h"].append(h_ne)
            psucc[b]["m"].append(m_ps); psucc[b]["h"].append(h_ps)
        print(f"  beta={b:5.1f} | P_succ over T={Ts}:")
        print("            " + "  ".join(f"{v:.3f}" for v in psucc[b]["m"]))

    # OAA rounds ~ (pi/4) / sqrt(P_succ) ; depth compounding P_succ^{-L/2}
    Ls = [1, 4, 12]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    for b in betas:
        m = np.array(psucc[b]["m"]); h = np.array(psucc[b]["h"])
        ax1.errorbar(Ts, m, yerr=h, marker="o", capsize=4, label=f"$\\beta$={b}")
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("Sequence length $T$"); ax1.set_ylabel("$P_{succ}=n_{eff}/T$")
    ax1.set_title("Post-selection success probability")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend()

    for b in betas:
        m = np.array(psucc[b]["m"])
        oaa = (np.pi / 4) / np.sqrt(m)
        ax2.plot(Ts, oaa, marker="s", label=f"$\\beta$={b}")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence length $T$")
    ax2.set_ylabel("OAA rounds $\\sim \\frac{\\pi}{4}P_{succ}^{-1/2}$")
    ax2.set_title("Amplitude-amplification overhead (per layer)")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend()

    # depth compounding at the sharpest beta
    b = betas[-1]
    m = np.array(psucc[b]["m"])
    for L in Ls:
        ax3.plot(Ts, m ** (-L / 2.0), marker="^", label=f"$L$={L} layers")
    ax3.set_xscale("log", base=2); ax3.set_yscale("log")
    ax3.set_xlabel("Sequence length $T$")
    ax3.set_ylabel("cumulative cost $\\sim P_{succ}^{-L/2}$")
    ax3.set_title(f"Depth compounding ($\\beta$={b})")
    ax3.grid(True, which="both", alpha=0.3); ax3.legend()

    fig.suptitle("Resource cost of the canonical coherent softmax construction "
                 "(upper bound on success / lower bound on cost for THIS route)",
                 fontsize=12)
    fig.savefig("figB_resource_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   -> saved figB_resource_cost.png")


# ======================================================================
# PART C  --  Coherent polynomial route: degree must scale with beta
# ======================================================================
def weights_classical(scores, beta):
    e = np.exp(beta * (scores - scores.max()))
    return e / e.sum()


def weights_poly_born(scores, degree, beta):
    """Born-rule readout: amplitude ~ P_d approximating exp((beta/2) x),
    weight = |P_d|^2 / sum|P_d|^2.  Non-negative by construction."""
    amp = chebyshev_exp(scores, degree, beta / 2.0)
    w = amp ** 2
    return w / w.sum()


def part_C():
    print("\n" + "=" * 70)
    print("PART C : degree-vs-beta scaling (Peak Flattening), Born readout")
    print("=" * 70)

    from scipy.stats import entropy

    # --- (1) minimal degree to reach KL target, as a function of beta ---
    betas = [1, 2, 3, 4, 5, 6, 8, 10]
    T, d_k, trials = 16, 4, 100
    kl_target = 1e-2
    max_deg = 40

    min_degree = []
    for b in betas:
        needed = []
        for _ in range(trials):
            scores = gram_scores(T, d_k, rng)[rng.integers(T)]
            pc = weights_classical(scores, b)
            dd = max_deg
            for deg in range(1, max_deg + 1):
                pq = weights_poly_born(scores, deg, b)
                if entropy(pc + 1e-12, pq + 1e-12) < kl_target:
                    dd = deg
                    break
            needed.append(dd)
        min_degree.append(np.mean(needed))
        print(f"  beta={b:4.1f} -> mean min degree for KL<{kl_target}: {min_degree[-1]:.1f}")

    # linear fit d ~ a*beta + c
    a, c = np.polyfit(betas, min_degree, 1)
    print(f"  linear fit: d ~ {a:.2f} * beta + {c:.2f}  (confirms d ~ O(beta))")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(betas, min_degree, "o-", lw=2, label="empirical")
    ax1.plot(betas, a * np.array(betas) + c, "--", color="gray",
             label=f"linear fit  $d\\approx{a:.1f}\\beta{c:+.1f}$")
    ax1.set_xlabel("softmax sharpness $\\beta$")
    ax1.set_ylabel(f"min degree $d$ for KL < {kl_target}")
    ax1.set_title("Coherent polynomial route: $d \\sim \\mathcal{O}(\\beta)$")
    ax1.grid(True, alpha=0.3); ax1.legend()

    # --- (2) Peak-Flattening heatmap at one (beta, T), Born readout ---
    rng2 = np.random.default_rng(7)
    Tg, beta_g = 32, 5.0
    K = rng2.standard_normal((Tg, d_k)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    tgt = 10
    Qv = K[tgt] + rng2.normal(0, 0.05, d_k); Qv /= np.linalg.norm(Qv)
    sc = K @ Qv
    rows = np.vstack([
        weights_classical(sc, beta_g),
        weights_poly_born(sc, 3, beta_g),
        weights_poly_born(sc, 7, beta_g),
    ])
    im = ax2.imshow(rows, cmap="inferno", aspect="auto")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["classical\n(ideal)", "poly $d{=}3$\n(flattened)",
                         "poly $d{=}7$\n(restored)"], fontsize=10)
    ax2.set_xlabel("token index")
    ax2.set_title(f"Peak flattening / recovery ($\\beta$={beta_g}, $T$={Tg})")
    import matplotlib.patches as patches
    ax2.add_patch(patches.Rectangle((tgt - 0.5, -0.5), 1, 3, lw=2,
                                     edgecolor="cyan", facecolor="none"))
    fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="attention weight")

    print(f"\n  target weight  classical : {rows[0, tgt]:.4f}")
    print(f"  target weight  poly d=3  : {rows[1, tgt]:.4f}")
    print(f"  target weight  poly d=7  : {rows[2, tgt]:.4f}")

    fig.savefig("figC_degree_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   -> saved figC_degree_scaling.png")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    print("\nDone. Figures: figA_nogo_three_operations.png, "
          "figB_resource_cost.png, figC_degree_scaling.png")
