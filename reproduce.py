#!/usr/bin/env python3
"""
reproduce.py -- single entry point for
  "Query Complexity of Coherent Softmax Attention and the Failure of the
   Singular-Value Route"

Design contract with the manuscript
-----------------------------------
1. Every function is named for the paper section or claim it supports, and
   its docstring cites the equation or statement it implements.
2. Everything quantitative that the paper asserts is either (a) proved in the
   paper and CHECKED here numerically, or (b) measured here and QUOTED there.
   The machine-readable results land in claims.json (checks, PASS/FAIL) and
   numbers.json (measured values).
3. These are classical numerical computations throughout. No quantum circuit
   is compiled or executed; gate synthesis, QSP phase angles and ancilla
   overhead are not modelled. See the Methods section of the paper.

Usage
-----
    python reproduce.py all        # verification suite + all figures (~4 min)
    python reproduce.py verify     # verification suite only        (~30 s)
    python reproduce.py fig2       # one figure
    python reproduce.py all --quick    # reduced trial counts, smoke test

Outputs: figures/Fig1..Fig5.{eps,pdf,png}, figures/numbers.json,
figures/claims.json. Fixed per-section seeds; reruns are bit-identical on a
fixed numpy version.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from scipy.special import ive

OUTDIR = os.environ.get("COH_OUTDIR", "figures")
BOOTSTRAP_B = 10000

SEEDS = {
    "fig1": 20260401, "fig2": 20260402, "fig3": 20260403,
    "fig4": 20260404, "fig5": 20260405, "verify": 20260406,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "0.88", "grid.alpha": 1.0,
    "grid.linewidth": 0.4,
})
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v"]

CLAIMS = {}   # name -> {"pass": bool, ...detail...}


def record(name, ok, **detail):
    CLAIMS[name] = {"pass": bool(ok), **detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


# ===========================================================================
# Methods: shared primitives (paper Section 3)
# ===========================================================================

def make_scores(T, dk, rng, normalize=None):
    """Q, K i.i.d. N(0,1); S = Q K^T / sqrt(dk). Methods, 'Score generation'."""
    Q = rng.standard_normal((T, dk))
    K = rng.standard_normal((T, dk))
    S = Q @ K.T / np.sqrt(dk)
    if normalize == "max":            # into [-1,1] for the QSP domain
        m = np.abs(S).max()
        if m > 0:
            S = S / m
    elif normalize == "spectral":     # Fig. 1 only
        S = S / np.linalg.norm(S, 2)
    return S


def softmax_rows(S, beta):
    """Row-wise, max-subtracted softmax; Eq. (softmax)."""
    Z = beta * S
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def n_eff_rows(S, beta):
    """n_eff per row; Eq. (neff)."""
    Z = beta * S
    Z = Z - Z.max(axis=1, keepdims=True)
    return np.exp(Z).sum(axis=1)


def cheb_exp_coeffs(beta_prime, degree):
    """
    Chebyshev coefficients of exp(beta' x) on [-1,1] truncated at `degree`:
    c_0 = I_0(beta'), c_k = 2 I_k(beta').  Evaluated via the exponentially
    scaled Bessel routine ive(k,b) = I_k(b) e^{-b} to avoid overflow, with
    the common factor restored (Methods, 'Coherent activation').
    """
    k = np.arange(degree + 1)
    c = 2.0 * ive(k, beta_prime)
    c[0] = ive(0, beta_prime)
    return c * np.exp(beta_prime)


def born_weights(scores, beta, degree):
    """Born readout: amplitude ~ P_d approximating e^{(beta/2)x}; weight
    proportional to |P_d|^2, renormalised over the row (Methods)."""
    amp = C.chebval(scores, cheb_exp_coeffs(beta / 2.0, degree))
    w = amp ** 2
    tot = w.sum()
    if tot <= 0 or not np.isfinite(tot):
        return np.full_like(scores, 1.0 / scores.size)
    return w / tot


def kl(p, q, eps=1e-300):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def bootstrap_ci(x, alpha=0.05, B=BOOTSTRAP_B, seed=0):
    """95% percentile bootstrap CI for the mean (Methods, 'Statistics')."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def ols_fit(x, y, B=2000, seed=0):
    """OLS with R^2 and bootstrap CI on the slope (Methods, 'Statistics')."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rng = np.random.default_rng(seed); slopes = []
    for _ in range(B):
        idx = rng.integers(0, x.size, size=x.size)
        if np.ptp(x[idx]) == 0:
            continue
        slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
    ci = tuple(np.percentile(slopes, [2.5, 97.5])) if slopes else (np.nan, np.nan)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "slope_ci95": [float(ci[0]), float(ci[1])]}


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUTDIR}/{name}.{{pdf,eps,png}}")


# ===========================================================================
# Section 5 verification: equivariance (Lemmas 1-2, witness)
# ===========================================================================

def sigma_o(S):
    """Element-wise exponential (Definition 2)."""
    return np.exp(S)


def svt(S, func):
    """Singular-value transformation (Definition 1)."""
    U, sig, Vt = np.linalg.svd(S)
    return (U * func(sig)) @ Vt


def verify_section5(rng):
    print("Section 5: equivariance claims")
    T = 5
    S = rng.standard_normal((T, T))

    # Lemma 1: SVT is equivariant under any orthogonal pair.
    def rand_orth(rng, T):
        Q, _ = np.linalg.qr(rng.standard_normal((T, T)))
        return Q
    A, B = rand_orth(rng, T), rand_orth(rng, T)
    lhs = svt(A @ S @ B.T, np.exp)
    rhs = A @ svt(S, np.exp) @ B.T
    record("lemma1_svt_biorthogonal_equivariance",
           np.allclose(lhs, rhs, atol=1e-9),
           max_dev=float(np.abs(lhs - rhs).max()))

    # Lemma 2 (forward): unsigned permutation pairs, and the global sign pair,
    # are equivariances of sigma_o.
    P1 = np.eye(T)[rng.permutation(T)]
    P2 = np.eye(T)[rng.permutation(T)]
    ok_perm = np.allclose(sigma_o(P1 @ S @ P2.T), P1 @ sigma_o(S) @ P2.T)
    ok_gsign = np.allclose(sigma_o((-P1) @ S @ (-P2).T),
                           (-P1) @ sigma_o(S) @ (-P2).T)
    record("lemma2_forward_permutations_and_global_sign", ok_perm and ok_gsign)

    # Lemma 2 (converse witnesses): a single sign flip fails; a rotation fails.
    D = np.diag([-1.0] + [1.0] * (T - 1))
    dev_sign = np.abs(sigma_o(D @ S) - D @ sigma_o(S)).max()
    th = np.pi / 5
    R = np.eye(T); R[:2, :2] = [[np.cos(th), -np.sin(th)],
                                 [np.sin(th), np.cos(th)]]
    dev_rot = np.abs(sigma_o(R @ S @ R.T) - R @ sigma_o(S) @ R.T).max()
    record("lemma2_converse_signflip_and_rotation_fail",
           dev_sign > 1e-6 and dev_rot > 1e-6,
           dev_single_sign_flip=float(dev_sign), dev_rotation=float(dev_rot))

    # Explicit witness (Section 5): exchange matrix, exact values.
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    ew = sigma_o(X)                       # [[1,e],[e,1]]
    sv = svt(X, np.exp)                   # e * X = [[0,e],[e,0]]
    ok = (np.allclose(ew, [[1, np.e], [np.e, 1]])
          and np.allclose(sv, [[0, np.e], [np.e, 0]], atol=1e-12))
    record("witness_2x2_exchange", ok,
           diagonal_discrepancy=float(np.abs(ew - sv)[0, 0]))

    # Hadamard-square intertwining used in the converse proof of Lemma 2:
    # (ASB^T)^{o2} = A S^{o2} B^T holds for permutation pairs, fails for R.
    ok_sq_perm = np.allclose((P1 @ S @ P2.T) ** 2, P1 @ (S ** 2) @ P2.T)
    dev_sq_rot = np.abs((R @ S @ R.T) ** 2 - R @ (S ** 2) @ R.T).max()
    record("lemma2_proof_hadamard_square_step",
           ok_sq_perm and dev_sq_rot > 1e-6, dev_rotation=float(dev_sq_rot))


# ===========================================================================
# Section 5.1 verification + Fig 5: Proposition 1 (approximation gap)
# ===========================================================================

def prop1_bound(a):
    """Right-hand side of Eq. (gap)."""
    T = a.size
    return np.sqrt(T * T - T) / np.sqrt(T * T - T + np.exp(2 * a).sum())


def verify_prop1(rng):
    print("Section 5.1: Proposition 1")
    T = 8
    a = rng.uniform(-1, 1, T)
    while len(np.unique(np.round(np.abs(a), 12))) < T:   # distinct |a_i| a.s.
        a = rng.uniform(-1, 1, T)

    target = np.ones((T, T)); np.fill_diagonal(target, np.exp(a))
    denom = np.linalg.norm(target, "fro")

    # (i) Lower bound: for arbitrary polynomials, the relative error is
    # >= prop1_bound(a). Check on random polynomials of several degrees.
    ok_lb, min_ratio = True, np.inf
    for d in (1, 3, 8, 20):
        for _ in range(20):
            coeffs = rng.standard_normal(d + 1)
            approx = svt(np.diag(a), lambda s: np.polyval(coeffs, s))
            ratio = np.linalg.norm(approx - target, "fro") / denom
            min_ratio = min(min_ratio, ratio)
            ok_lb &= ratio >= prop1_bound(a) - 1e-12

    # (ii) Attainment: the interpolating polynomial P(|a_i|) = sign(a_i)e^{a_i}
    # achieves the bound exactly when the |a_i| are distinct.
    nodes = np.abs(a)
    vals = np.sign(a) * np.exp(a)
    interp = np.polyfit(nodes, vals, T - 1)
    attained = np.linalg.norm(svt(np.diag(a), lambda s: np.polyval(interp, s))
                              - target, "fro") / denom
    ok_eq = abs(attained - prop1_bound(a)) < 1e-8
    record("prop1_lower_bound_and_attainment", ok_lb and ok_eq,
           bound=float(prop1_bound(a)), attained=float(attained),
           min_random_poly_ratio=float(min_ratio))


def fig5(rng, Ts=(2, 4, 8, 16, 32, 64, 128), degrees=(2, 4, 8, 16, 32),
         trials=200):
    """Fig 5: the degree-independent gap of Proposition 1."""
    print("Fig 5: approximation gap")
    Ts = np.asarray(Ts, int)
    best = np.zeros(len(Ts))
    emp = {d: np.zeros(len(Ts)) for d in degrees}
    for i, T in enumerate(Ts):
        b_acc = []; e_acc = {d: [] for d in degrees}
        for _ in range(trials):
            a = rng.uniform(-1, 1, T)
            target = np.ones((T, T)); np.fill_diagonal(target, np.exp(a))
            denom = np.linalg.norm(target, "fro")
            b_acc.append(prop1_bound(a))
            for d in degrees:
                c = cheb_exp_coeffs(1.0, d)     # P_d ~ exp, unit beta'
                approx = svt(np.diag(a), lambda s: C.chebval(s, c))
                e_acc[d].append(np.linalg.norm(approx - target, "fro") / denom)
        best[i] = np.mean(b_acc)
        for d in degrees:
            emp[d][i] = np.mean(e_acc[d])

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.6))
    ax = axes[0]
    for i, d in enumerate(degrees):
        ax.plot(Ts, emp[d], marker=MARKERS[i % 5], ms=3.5, lw=1.2,
                color=PALETTE[i % 5], label=f"degree {d}")
    ax.plot(Ts, best, "k--", lw=1.2, label="optimal $P$ (any degree)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel(r"$\|\mathrm{SVT}_P(S)-\sigma_\circ(S)\|_F/\|\sigma_\circ(S)\|_F$")
    ax.set_title("relative gap on diagonal inputs")
    ax.set_ylim(0, 1.45)
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    ax.text(0.03, 0.42, "all degrees coincide", transform=ax.transAxes,
            fontsize=6.5, color="0.35")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    ax.plot(Ts, 1.0 - best, "k-", marker="o", ms=3.5, lw=1.2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel(r"$1-$ relative gap")
    ax.set_title(r"gap $\to 1$: no degree closes it")
    fig.tight_layout()
    save(fig, "Fig5")
    return {"Ts": Ts.tolist(), "optimal_relative_gap": best.tolist(),
            "empirical_by_degree": {str(d): emp[d].tolist() for d in degrees},
            "trials": trials}


# ===========================================================================
# Section 6 verification: hard family (Lemma 4) and reduction arithmetic
# ===========================================================================

def verify_section6():
    """
    Lemma 4 (hard family): with M scores at 0 and T-M at -Delta,
    beta*Delta = ln(100 T)  (the s_max = 0 convention of Section 6.1):
      M <= n_eff <= M + 1/100   and   marked mass >= 1 - 1/(100 M).
    """
    print("Section 6: hard instance family")
    rows, ok = [], True
    for T in (16, 64, 256, 1024, 4096):
        bd = np.log(100.0 * T)
        for M in (1, 2, max(1, int(np.sqrt(T))), T // 4, T // 2):
            s = np.full(T, -bd); s[:M] = 0.0          # s_max = 0 by promise
            z = s - s.max()
            n_eff = float(np.exp(z).sum())
            w = np.exp(z) / np.exp(z).sum()
            marked = float(w[:M].sum())
            ok &= (M <= n_eff <= M + 0.01 + 1e-12)
            ok &= (marked >= 1 - 1 / (100 * M) - 1e-12)
            rows.append({"T": T, "M": M, "n_eff": n_eff,
                         "marked_mass": marked,
                         "sqrt_T_over_neff": float(np.sqrt(T / n_eff))})
    record("lemma3_hard_family_bounds", ok, n_rows=len(rows))
    return rows


# ===========================================================================
# Section 7.1: truncation lemma (Lemma 5) + Fig 2 (degree scaling)
# ===========================================================================

def truncation_certificate(beta_prime, degree):
    """
    Lemma 5 certificate: for degree d >= beta',
        sup_{[-1,1]} |e^{beta' x} - P_d(x)|  <=  4 I_{d+1}(beta').
    Returns the certificate value 4 I_{d+1}(beta').
    """
    return float(4.0 * ive(degree + 1, beta_prime) * np.exp(beta_prime))


def verify_truncation_lemma():
    print("Section 7.1: Lemma 5 (truncation certificate)")
    x = np.linspace(-1, 1, 20001)
    ok, ok2, worst = True, True, -np.inf
    for bp in (0.5, 1.0, 2.5, 5.0):
        exact = np.exp(bp * x)
        d0 = int(np.ceil(bp))
        for d in range(d0, d0 + 25):
            err = np.abs(exact - C.chebval(x, cheb_exp_coeffs(bp, d))).max()
            cert = truncation_certificate(bp, d)
            ok &= err <= cert + 1e-12
            worst = max(worst, err - cert)
            # closed-form majorant of the certificate (Lemma 5(iii)):
            # I_{d+1}(bp) <= e^{bp/4} (e*bp/(2(d+1)))^{d+1}
            closed = np.exp(bp / 4.0) * (np.e * bp / (2 * (d + 1))) ** (d + 1)
            ok2 &= ive(d + 1, bp) * np.exp(bp) <= closed + 1e-15
    record("lemma5_sup_error_leq_certificate", ok, worst_margin=float(worst))
    record("lemma5_certificate_leq_closed_form", ok2)


def min_degree_kl(s_row, beta, threshold, max_degree=64):
    """Smallest d with KL(exact softmax || Born-readout weights) < threshold.
    Returns NaN if max_degree is insufficient (never silently capped)."""
    exact = softmax_rows(s_row[None, :], beta)[0]
    for d in range(1, max_degree + 1):
        if kl(exact, born_weights(s_row, beta, d)) < threshold:
            return d
    return np.nan


def fig2(rng, T=16, dk=4, trials=100, betas=(1, 2, 3, 4, 5, 6, 8, 10),
         thresholds=(1e-1, 1e-2, 1e-3)):
    print("Fig 2: degree scaling")
    betas = np.asarray(betas, float)
    D = {th: np.full((len(betas), trials), np.nan) for th in thresholds}
    censored = 0
    for ti in range(trials):
        row = make_scores(T, dk, rng, normalize="max")[0]
        for bi, b in enumerate(betas):
            for th in thresholds:
                v = min_degree_kl(row, b, th)
                if np.isnan(v):
                    censored += 1
                D[th][bi, ti] = v

    fits, means, cis = {}, {}, {}
    for th in thresholds:
        m = np.nanmean(D[th], axis=1)
        means[th] = m
        cis[th] = np.array([bootstrap_ci(D[th][bi], seed=1000 + bi)
                            for bi in range(len(betas))])
        fits[str(th)] = ols_fit(betas, m, seed=7)

    # Lemma 5 certificate curve: minimal d with 4 I_{d+1}(beta/2) e^{beta/2}
    # ... i.e. amplitude sup-error certificate <= 1e-2.
    cert_curve = []
    for b in betas:
        bp = b / 2.0
        d = int(np.ceil(bp))
        while truncation_certificate(bp, d) > 1e-2:
            d += 1
        cert_curve.append(d)
    cert_fit = ols_fit(betas, cert_curve, seed=9)

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.6))
    ax = axes[0]
    th = 1e-2
    ax.errorbar(betas, means[th],
                yerr=[means[th] - cis[th][:, 0], cis[th][:, 1] - means[th]],
                marker="o", ms=3.5, lw=1.2, color=PALETTE[0], capsize=2,
                label=r"empirical (KL $<10^{-2}$)")
    f = fits[str(th)]
    ax.plot(betas, f["slope"] * betas + f["intercept"], "--", lw=1.0,
            color="0.35",
            label=rf"fit $d={f['slope']:.2f}\beta{f['intercept']:+.2f}$,"
                  rf" $R^2={f['r2']:.3f}$")
    ax.plot(betas, cert_curve, ":", marker="x", ms=4, lw=1.0, color=PALETTE[3],
            label=r"Lemma 5 certificate ($\varepsilon=10^{-2}$)")
    ax.set_xlabel(r"softmax sharpness $\beta$")
    ax.set_ylabel("minimal degree $d$")
    ax.set_title(r"coherent route: $d\sim\mathcal{O}(\beta)$")
    ax.legend(frameon=False)
    ax = axes[1]
    for i, th in enumerate(thresholds):
        f = fits[str(th)]
        ax.errorbar(betas, means[th],
                    yerr=[means[th] - cis[th][:, 0], cis[th][:, 1] - means[th]],
                    marker=MARKERS[i], ms=3.5, lw=1.2, color=PALETTE[i],
                    capsize=2,
                    label=rf"KL $<10^{{{int(np.log10(th))}}}$: "
                          rf"slope {f['slope']:.2f} "
                          rf"[{f['slope_ci95'][0]:.2f},{f['slope_ci95'][1]:.2f}]")
    ax.set_xlabel(r"softmax sharpness $\beta$")
    ax.set_ylabel("minimal degree $d$")
    ax.set_title("linear scaling is threshold-independent")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "Fig2")
    return {"fits": fits, "certificate_curve": cert_curve,
            "certificate_fit": cert_fit, "censored_cells": censored,
            "T": T, "dk": dk, "trials": trials, "betas": betas.tolist()}


# ===========================================================================
# Section 7.2 verification: the canonical construction, and Fig 3
# ===========================================================================

def verify_construction(rng):
    """
    Exact statevector check of the canonical construction of Section 7.2
    under the s_max = 0 convention (Section 6.1), where c = 1:
    flag-qubit amplitudes cos(theta_j) = e^{beta s_j / 2} on the uniform
    superposition. Verifies (a) P(flag) = n_eff / T to machine precision,
    (b) the post-selected state equals |psi_s> to machine precision, and
    (c) with the degree-d polynomial in place of the exact exponential the
    post-selected distribution reproduces the Fig 2 pipeline.
    """
    print("Section 7.2: statevector check of the canonical construction")
    ok_p, ok_f, ok_poly = True, True, True
    for T in (4, 16, 64):
        s = make_scores(T, 4, rng, normalize="max")[0]
        s = s - s.max()                                  # promise: s_max = 0
        for beta in (1.0, 5.0, 10.0):
            amp_keep = np.exp(beta * s / 2.0) / np.sqrt(T)        # flag |0>
            amp_drop = np.sqrt(np.maximum(0.0, 1.0 / T - amp_keep ** 2))
            psucc = float((amp_keep ** 2).sum())
            ok_p &= abs(psucc - n_eff_rows(s[None, :], beta)[0] / T) < 1e-12
            post = amp_keep / np.linalg.norm(amp_keep)
            target = np.sqrt(softmax_rows(s[None, :], beta)[0])
            ok_f &= abs(1.0 - float(post @ target)) < 1e-12
            # norm bookkeeping of the dilation
            ok_f &= abs((amp_keep ** 2).sum() + (amp_drop ** 2).sum() - 1) < 1e-12
            # degree-d polynomial route matches the Fig 2 pipeline
            d = 7
            amp_poly = C.chebval(s, cheb_exp_coeffs(beta / 2.0, d))
            w_post = amp_poly ** 2 / (amp_poly ** 2).sum()
            ok_poly &= np.allclose(w_post, born_weights(s, beta, d))
    record("construction_psucc_equals_neff_over_T", ok_p)
    record("construction_postselected_state_is_psi_s", ok_f)
    record("construction_poly_route_matches_fig2_pipeline", ok_poly)


def fig3(rng, dk=4, trials=300, Ts=(4, 8, 16, 32, 64, 128),
         betas=(1.0, 2.0, 5.0, 10.0)):
    print("Fig 3: per-layer resource cost")
    Ts = np.asarray(Ts, int)
    P = np.zeros((len(betas), len(Ts)))
    Pci = np.zeros((len(betas), len(Ts), 2))
    for ti_, T in enumerate(Ts):
        per = {b: [] for b in betas}
        for _ in range(trials):
            S = make_scores(T, dk, rng)
            for b in betas:
                per[b].append(float(np.mean(n_eff_rows(S, b) / T)))
        for bi, b in enumerate(betas):
            arr = np.array(per[b])
            P[bi, ti_] = arr.mean()
            Pci[bi, ti_] = bootstrap_ci(arr, seed=2000 + ti_ * 10 + bi)

    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.5))
    ax = axes[0]
    for bi, b in enumerate(betas):
        ax.errorbar(Ts, P[bi], yerr=[P[bi] - Pci[bi, :, 0],
                                     Pci[bi, :, 1] - P[bi]],
                    marker=MARKERS[bi], ms=3.5, lw=1.2, color=PALETTE[bi],
                    capsize=2, label=rf"$\beta={b:g}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel(r"$P_{\mathrm{succ}}=n_{\mathrm{eff}}/T$")
    ax.set_title("post-selection success")
    ax.legend(frameon=False)
    ax = axes[1]
    for bi, b in enumerate(betas):
        ax.plot(Ts, (np.pi / 4) * P[bi] ** -0.5, marker=MARKERS[bi], ms=3.5,
                lw=1.2, color=PALETTE[bi], label=rf"$\beta={b:g}$")
    ax.plot(Ts, (np.pi / 4) * np.sqrt(Ts), ":", lw=1.0, color="0.4",
            label=r"$\Theta(\sqrt{T})$ (Thm 2, sharp limit)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel(r"AA rounds $\sim\frac{\pi}{4}P_{\mathrm{succ}}^{-1/2}$")
    ax.set_title("per-layer overhead")
    ax.legend(frameon=False)
    ax = axes[2]
    bi = len(betas) - 1
    for li, L in enumerate((1, 4, 12)):
        ax.plot(Ts, P[bi] ** (-L / 2.0), marker=MARKERS[li], ms=3.5, lw=1.2,
                color=PALETTE[li], label=f"L={L} layers")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length $T$")
    ax.set_ylabel(r"cumulative $P_{\mathrm{succ}}^{-L/2}$")
    ax.set_title(rf"fixed-sharpness envelope ($\beta={betas[bi]:g}$)")
    ax.legend(frameon=False)
    save(fig, "Fig3")
    return {"Ts": Ts.tolist(), "betas": list(betas),
            "P_succ_mean": P.tolist(), "P_succ_ci95": Pci.tolist(),
            "trials": trials, "dk": dk}


# ===========================================================================
# Section 7.3: depth compounding, Fig 4
# ===========================================================================

def rownorm(X, dk):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm * np.sqrt(dk)


def stack_run(T, dk, beta, L, rng, residual, project=True):
    """
    One stacked-attention trajectory (Methods, 'Stacked-layer experiment').
    project=True draws independent Wq, Wk, Wv ~ N(0, 1/dk) each layer.
    project=False (robustness check only) sets Q=K=V=X, which forces a
    maximal score diagonal and artificially sharp attention.
    """
    X = rownorm(rng.standard_normal((T, dk)), dk)
    out = []
    for _ in range(L):
        if project:
            Wq = rng.standard_normal((dk, dk)) / np.sqrt(dk)
            Wk = rng.standard_normal((dk, dk)) / np.sqrt(dk)
            Wv = rng.standard_normal((dk, dk)) / np.sqrt(dk)
            Q, K, V = X @ Wq, X @ Wk, X @ Wv
        else:
            Q = K = V = X
        S = Q @ K.T / np.sqrt(dk)
        out.append(float(np.mean(n_eff_rows(S, beta) / T)))
        W = softmax_rows(S, beta)
        Z = W @ V
        X = rownorm(X + Z, dk) if residual else rownorm(Z, dk)
    return np.array(out)


def fig4(rng, T=32, dk=4, L=12, trials=200, betas=(2.0, 5.0, 10.0)):
    print("Fig 4: depth compounding")
    layers = np.arange(1, L + 1)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.6), sharey=True)
    for ax, residual, name in zip(axes, (False, True),
                                  ("pure pooling", "residual")):
        results[name] = {}
        for bi, b in enumerate(betas):
            runs = np.array([stack_run(T, dk, b, L, rng, residual, True)
                             for _ in range(trials)])
            mean = runs.mean(axis=0)
            ci = np.array([bootstrap_ci(runs[:, l], seed=3000 + l)
                           for l in range(L)])
            ax.errorbar(layers, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean],
                        marker=MARKERS[bi], ms=3.5, lw=1.2, color=PALETTE[bi],
                        capsize=2, label=rf"$\beta={b:g}$")
            cum = np.array([np.prod(r ** -0.5) for r in runs])  # per trial
            results[name][f"beta={b:g}"] = {
                "per_layer_mean_Psucc": mean.tolist(),
                "per_layer_ci95": ci.tolist(),
                "realized_cumulative_amplification": float(cum.mean()),
                "realized_cumulative_ci95": list(bootstrap_ci(cum, seed=4242)),
                "fixed_sharpness_extrapolation": float(mean[0] ** (-L / 2.0)),
                "overstatement_factor": float(mean[0] ** (-L / 2.0) / cum.mean()),
            }
        ax.set_xlabel(r"layer index $\ell$")
        ax.set_title(rf"{name}  ($T={T}$, $d_k={dk}$)")
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"mean $n_{\mathrm{eff}}/T$  $(=P_{\mathrm{succ}})$")
    save(fig, "Fig4")

    # robustness check: unprojected variant, reported but not plotted
    results["_unprojected_robustness"] = {}
    for residual, name in ((False, "pure pooling"), (True, "residual")):
        for b in betas:
            runs = np.array([stack_run(T, dk, b, L, rng, residual, False)
                             for _ in range(max(50, trials // 4))])
            cum = np.array([np.prod(r ** -0.5) for r in runs])
            results["_unprojected_robustness"][f"{name}|beta={b:g}"] = {
                "layer1_Psucc": float(runs.mean(axis=0)[0]),
                "layer12_Psucc": float(runs.mean(axis=0)[-1]),
                "realized_cumulative_amplification": float(cum.mean()),
            }
    results["_config"] = {"T": T, "dk": dk, "L": L, "trials": trials,
                          "betas": list(betas)}
    return results


# ===========================================================================
# Fig 1: the three operations
# ===========================================================================

def fig1(rng, T=8, dk=4):
    print("Fig 1: three operations")
    S = make_scores(T, dk, rng, normalize="spectral")
    lam, V = np.linalg.eig(S)
    mats = [np.exp(S),
            np.real(V @ np.diag(np.exp(lam)) @ np.linalg.inv(V)),
            svt(S, np.exp)]
    titles = [r"element-wise $[\exp(S)]_{ij}$" + "\n(softmax needs this)",
              r"matrix exp $\exp(S)$" + "\n(eigenvalue calculus)",
              r"QSVT  $U e^{\Sigma} V^{\dagger}$" + "\n(what QSVT does)"]
    vmax = max(np.abs(m).max() for m in mats)
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.5))
    for ax, M, t in zip(axes, mats, titles):
        im = ax.imshow(M, cmap="viridis", vmin=-vmax, vmax=vmax)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    save(fig, "Fig1")
    return {"max_abs_diff_elementwise_vs_svt":
            float(np.abs(mats[0] - mats[2]).max())}


# ===========================================================================
# driver
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="all",
                    choices=["all", "verify", "fig1", "fig2", "fig3",
                             "fig4", "fig5"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    global OUTDIR
    if args.outdir:
        OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    tr = dict(fig2=20, fig3=30, fig4=20, fig5=50) if args.quick else \
         dict(fig2=100, fig3=300, fig4=200, fig5=200)

    numbers = {"_provenance": {
        "seeds": SEEDS, "bootstrap_B": BOOTSTRAP_B,
        "numpy": np.__version__, "python": sys.version.split()[0],
        "note": ("classical numerical computation of score statistics; "
                 "no quantum circuit is compiled or executed")}}

    if args.target in ("all", "verify"):
        vr = np.random.default_rng(SEEDS["verify"])
        verify_section5(vr)
        verify_prop1(vr)
        numbers["hard_family"] = verify_section6()
        verify_truncation_lemma()
        verify_construction(vr)

    if args.target in ("all", "fig1"):
        numbers["fig1"] = fig1(np.random.default_rng(SEEDS["fig1"]))
    if args.target in ("all", "fig2"):
        numbers["fig2"] = fig2(np.random.default_rng(SEEDS["fig2"]),
                               trials=tr["fig2"])
    if args.target in ("all", "fig3"):
        numbers["fig3"] = fig3(np.random.default_rng(SEEDS["fig3"]),
                               trials=tr["fig3"])
    if args.target in ("all", "fig4"):
        numbers["fig4"] = fig4(np.random.default_rng(SEEDS["fig4"]),
                               trials=tr["fig4"])
    if args.target in ("all", "fig5"):
        numbers["fig5"] = fig5(np.random.default_rng(SEEDS["fig5"]),
                               trials=tr["fig5"])

    # merge rather than clobber on partial runs
    npath = os.path.join(OUTDIR, "numbers.json")
    if os.path.exists(npath):
        with open(npath) as f:
            old = json.load(f)
        old.update(numbers); numbers = old
    with open(npath, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"  wrote {npath}")

    if CLAIMS:
        cpath = os.path.join(OUTDIR, "claims.json")
        with open(cpath, "w") as f:
            json.dump(CLAIMS, f, indent=2)
        n_ok = sum(1 for v in CLAIMS.values() if v["pass"])
        print(f"  wrote {cpath}  ({n_ok}/{len(CLAIMS)} checks passed)")
        if n_ok < len(CLAIMS):
            sys.exit(1)


if __name__ == "__main__":
    main()
