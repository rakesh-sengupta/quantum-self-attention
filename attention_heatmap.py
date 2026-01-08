import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
import matplotlib.patches as patches

# ==========================================
# 1. Setup & Functions
# ==========================================
np.random.seed(42)  # Fixed seed

def chebyshev_expansion(x, degree, beta):
    approx = iv(0, beta)
    T_prev, T_curr = 1.0, x
    for k in range(1, degree + 1):
        term = 2 * iv(k, beta) * T_curr
        approx += term
        T_next = 2 * x * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
    return approx

def get_weights_classical(scores, beta=1.0):
    e_x = np.exp(beta * (scores - np.max(scores)))
    return e_x / e_x.sum()

def get_weights_quantum(scores, degree, beta=1.0):
    weights = np.array([chebyshev_expansion(s, degree, beta) for s in scores])
    return np.abs(weights) / np.sum(np.abs(weights))

# ==========================================
# 2. Generate Data (T=32, d=4)
# ==========================================
T = 32
d = 4
beta = 5.0 

# Create vectors
K = np.random.randn(T, d)
K = K / np.linalg.norm(K, axis=1, keepdims=True)

# Target at index 10
target_idx = 10
Q = K[target_idx] + np.random.normal(0, 0.05, d)
Q = Q / np.linalg.norm(Q)
scores = np.dot(K, Q)

# ==========================================
# 3. Calculate Weights for Comparison
# ==========================================
w_classical = get_weights_classical(scores, beta=beta)
w_quantum_low = get_weights_quantum(scores, degree=3, beta=beta)  # The Failure
w_quantum_high = get_weights_quantum(scores, degree=7, beta=beta) # The Fix

# ==========================================
# 4. Plotting the 3-Row Heatmap
# ==========================================
fig, ax = plt.subplots(figsize=(12, 5))

# Stack: Classical, Low Degree, High Degree
data = np.vstack([w_classical, w_quantum_low, w_quantum_high])

# Create Heatmap
im = ax.imshow(data, cmap='inferno', aspect='auto', interpolation='nearest')

# Formatting
ax.set_yticks([0, 1, 2])
ax.set_yticklabels([
    'Classical\n(Ideal)', 
    'Quantum $d=3$\n(Flattened)', 
    'Quantum $d=7$\n(Restored)'
], fontsize=11)

ax.set_xlabel('Token Index', fontsize=12)
ax.set_title(f'Peak Flattening & Recovery: Impact of Polynomial Degree ($\\beta={beta}$)', fontsize=14)

# Colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Attention Weight', rotation=270, labelpad=15)

# Highlight Target
rect = patches.Rectangle((target_idx - 0.5, -0.5), 1, 3, linewidth=2, edgecolor='cyan', facecolor='none')
ax.add_patch(rect)
ax.text(target_idx, -0.8, 'Target', color='cyan', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('attention_heatmap_comparison.png', dpi=300)
plt.show()

# ==========================================
# 5. Analysis Output
# ==========================================
print(f"Target Weight (Classical): {w_classical[target_idx]:.4f}")
print(f"Target Weight (Poly d=3):  {w_quantum_low[target_idx]:.4f} (Drop: {w_classical[target_idx]-w_quantum_low[target_idx]:.4f})")
print(f"Target Weight (Poly d=7):  {w_quantum_high[target_idx]:.4f} (Recovery!)")
