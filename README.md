# quantum-self-attention

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
