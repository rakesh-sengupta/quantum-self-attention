# QIP submission package

**Query Complexity of Coherent Softmax Attention and the Failure of the Singular-Value Route**
Rakesh Sengupta — submission to *Quantum Information Processing* (Springer).

The manuscript carries full proofs of every stated result, and `reproduce.py`
implements exactly what the Methods section says, verifies every checkable
claim, and emits every number the paper quotes.

## Contents

| File | Purpose |
|---|---|
| `Fig1–Fig5.{eps,pdf}` | Figures (EPS for Springer, PDF for local preview). |
| `reproduce.py` | Single entry point: verification suite + all figures + all numbers. |
| `numbers.json` | Machine-readable record of every quoted value, with seeds. |
| `claims.json` | Pass/fail record of the 12 verification checks (all pass). |
| `requirements.txt` | numpy, scipy, matplotlib. |

## Reproducing everything

```bash
pip install -r requirements.txt
python reproduce.py verify        # ~30 s: 12 numerical checks of the lemmas
python reproduce.py all           # ~4 min: checks + all five figures + numbers
```

`reproduce.py all` exits nonzero if any check fails. Fixed per-figure seeds;
reruns are bit-identical on a fixed numpy version. After regenerating, copy
`figures/Fig*.{eps,pdf}`, `figures/numbers.json` and `figures/claims.json`
into the package root.

What the verification suite checks, by paper statement:

- **Lemma 1** — SVT bi-orthogonal equivariance on random orthogonal pairs.
- **Lemma 2** — forward direction (permutations, global sign pair); converse
  witnesses (single sign flip and rotation both fail); and the Hadamard-square
  identity (Eq. 10) on which the converse proof rests.
- **Witness** — the 2×2 exchange-matrix computation, exact values.
- **Proposition 1** — the lower bound against random polynomials of four
  degrees, and exact attainment by the interpolating polynomial.
- **Lemma 3** — hard-family constants for T up to 4096, M up to T/2.
- **Lemma 5** — both inequalities of the truncation certificate on a dense grid.
- **Construction (§7.2)** — exact statevector computation of the flag-qubit
  dilation: P_succ = n_eff/T and post-selected state = |ψ_s⟩ to machine
  precision; polynomial variant reproduces the Fig. 2 pipeline.

## Notes on the simulation design

Scores follow the manuscript's stated model, s = ⟨q,k⟩/√d_k with Gaussian
Q, K. The stacked-layer experiment draws independent per-layer W_q, W_k, W_v;
with Q = K = V = X and row-normalised X the score matrix carries a constant
maximal diagonal, so every token self-attends maximally and attention looks
artificially sharp — the unprojected variant is retained in `numbers.json`
only as a robustness check. Cumulative amplification factors are averaged per
trial, never taken as a product of per-layer means. Minimal-degree searches
return a missing value rather than a capped one when the search range is
exhausted; censoring counts are recorded and are zero at all reported
parameters.
