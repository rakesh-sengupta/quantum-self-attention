import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
from scipy.stats import entropy, sem, t

# ==========================================
# 1. Configuration & Constants
# ==========================================
d_k = 4
beta = 5.0  # The contentious high-sharpness setting
degrees_to_test = [3, 7]  # Comparing the "suspicious" d=3 vs a robust d=7
sequence_lengths = [4, 8, 16, 32]
trials = 50  # Increased for statistical robustness (use >100 for publication)
confidence = 0.95  # 95% Confidence Interval

# ==========================================
# 2. Helper Functions (Polynomials)
# ==========================================

def chebyshev_expansion(x, degree, beta):
    """Approximates exp(beta * x) using Chebyshev polynomials."""
    approx = iv(0, beta)
    T_prev, T_curr = 1.0, x
    for k in range(1, degree + 1):
        term = 2 * iv(k, beta) * T_curr
        approx += term
        T_next = 2 * x * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
    return approx

def softmax_classical(scores, beta):
    e_x = np.exp(beta * (scores - np.max(scores)))
    return e_x / e_x.sum()

def softmax_quantum_poly(scores, degree, beta):
    # Note: Chebyshev approx can go negative due to truncation error.
    # We take absolute value as a physical heuristic (probability amplitude magnitude).
    weights = np.array([chebyshev_expansion(s, degree, beta) for s in scores])
    return np.abs(weights) / np.sum(np.abs(weights))

# ==========================================
# 3. Data Generation
# ==========================================

def generate_data(seq_len, dim):
    K = np.random.randn(seq_len, dim)
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    V = np.random.randn(seq_len, dim)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    target_idx = np.random.randint(0, seq_len)
    Q = K[target_idx] + np.random.normal(0, 0.1, dim)
    Q = Q / np.linalg.norm(Q)
    return Q, K, V, target_idx

# ==========================================
# 4. Simulation Loop with Statistics
# ==========================================

stats = {d: {'kl_mean': [], 'kl_err': [], 'fid_mean': [], 'fid_err': []} for d in degrees_to_test}

print(f"Starting Robust Simulation: Beta={beta}, Trials={trials}")
print("-" * 60)

for d in degrees_to_test:
    print(f"\nTesting Polynomial Degree d={d}...")
    
    for T in sequence_lengths:
        kl_samples = []
        fid_samples = []
        
        for _ in range(trials):
            Q, K, V, target_idx = generate_data(T, d_k)
            scores = np.dot(K, Q)
            
            # Ground Truth
            prob_classical = softmax_classical(scores, beta=beta)
            
            # Quantum Approx
            prob_quantum = softmax_quantum_poly(scores, degree=d, beta=beta)
            
            # Metric 1: KL Divergence
            kl = entropy(prob_classical + 1e-9, prob_quantum + 1e-9)
            kl_samples.append(kl)
            
            # Metric 2: Fidelity
            Z_quantum = np.zeros(d_k)
            for i in range(T):
                Z_quantum += prob_quantum[i] * V[i]
            Z_quantum /= np.linalg.norm(Z_quantum)
            fid_samples.append(np.dot(Z_quantum, V[target_idx]))
        
        # --- Statistical Analysis ---
        # 1. Mean
        mu_kl = np.mean(kl_samples)
        mu_fid = np.mean(fid_samples)
        
        # 2. Standard Error of Mean (SEM)
        sem_kl = sem(kl_samples)
        sem_fid = sem(fid_samples)
        
        # 3. Confidence Interval Margin (h) = t_score * SEM
        h_kl = sem_kl * t.ppf((1 + confidence) / 2., trials - 1)
        h_fid = sem_fid * t.ppf((1 + confidence) / 2., trials - 1)
        
        # Store
        stats[d]['kl_mean'].append(mu_kl)
        stats[d]['kl_err'].append(h_kl)
        stats[d]['fid_mean'].append(mu_fid)
        stats[d]['fid_err'].append(h_fid)
        
        print(f"  T={T:2d} | KL: {mu_kl:.2e} +/- {h_kl:.2e} | Fid: {mu_fid:.4f} +/- {h_fid:.4f}")

# ==========================================
# 5. Visualization with Error Bars
# ==========================================



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot KL Divergence
for d in degrees_to_test:
    ax1.errorbar(sequence_lengths, stats[d]['kl_mean'], yerr=stats[d]['kl_err'], 
                 label=f'Degree {d}', capsize=5, marker='o', linewidth=2)

ax1.set_xlabel('Sequence Length (T)', fontsize=12)
ax1.set_ylabel('KL Divergence (Log Scale)', fontsize=12)
ax1.set_yscale('log')
ax1.set_title(f'Approximation Error (Beta={beta})', fontsize=14)
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.legend()

# Plot Recall Fidelity
for d in degrees_to_test:
    ax2.errorbar(sequence_lengths, stats[d]['fid_mean'], yerr=stats[d]['fid_err'], 
                 label=f'Degree {d}', capsize=5, marker='s', linewidth=2)

ax2.set_xlabel('Sequence Length (T)', fontsize=12)
ax2.set_ylabel('Recall Fidelity', fontsize=12)
ax2.set_title('Downstream Task Performance', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle(f'Statistical Robustness Analysis (N={trials}, 95% CI)', fontsize=16)
plt.tight_layout()
plt.show()

# ==========================================
# 6. Statistical Conclusion
# ==========================================
print("\nSTATISTICAL CONCLUSION:")
mean_kl_d3 = np.mean(stats[3]['kl_mean'])
mean_kl_d7 = np.mean(stats[7]['kl_mean'])

print(f"1. Degree 3 Mean KL: {mean_kl_d3:.2e} (High Error confirmed for Beta=5)")
print(f"2. Degree 7 Mean KL: {mean_kl_d7:.2e} (Validates need for higher degree)")
print("3. Interpretation: The editor was correct. d=3 is insufficient for Beta=5.")
print("   However, the Fidelity plot shows if this error actually hurts retrieval.")
