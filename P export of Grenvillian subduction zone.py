import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Normal parameters ========== #
N = 1000000  # Samplings
rho_crust = 3300.0                      # kg/m^3
w_mol_P2O5 = 141.94                # g/mol
p2o5_weight_fraction = 0.005  # 0.5 wt%

# ========== 2. Define parameter ========== #

def sample_L(n, low=1.0e4, high=5.0e6):
    return np.random.uniform(low, high, n)

def sample_L_fixed(n, L_value=5.0e6):
    return np.full(n, L_value, dtype=float)

def sample_S_triangular(n):
    return np.random.triangular(0.028, 0.09, 0.102, n)

def sample_D(n):
    return np.random.uniform(1000, 2000, n)

def sample_fraction(n, low, high):
    return np.random.uniform(low, high, n)

# ========== 3. Parameter assignment ========== #

# Left: random L
L_vals_left = sample_L(N)
S_vals_left = sample_S_triangular(N)
D_vals_left = sample_D(N)

Hs_left = sample_fraction(N, 0.5, 0.8)
phi_s_left = sample_fraction(N, 0.12, 0.3)
P_alt_left = sample_fraction(N, 0.6, 1.0)
Lp2o5_left = sample_fraction(N, 0.8, 0.9)
c_fixed_left = sample_fraction(N, 0.0, 1.0)

# Right: fixed L
L_vals_right = sample_L_fixed(N, L_value=5.0e6)
S_vals_right = sample_S_triangular(N)
D_vals_right = sample_D(N)

Hs_right = sample_fraction(N, 0.5, 0.8)
phi_s_right = sample_fraction(N, 0.12, 0.3)
P_alt_right = sample_fraction(N, 0.6, 1.0)
Lp2o5_right = sample_fraction(N, 0.8, 0.9)
c_fixed_right = sample_fraction(N, 0.0, 1.0)

# ========== 4. Flux computation function ========== #

def compute_flux(L, S, D, Hs, phi_s, P_alt, Lp2o5, c_fixed):
    v = L * S * D * Hs    # Volume (m^3/yr)
    m = v * rho_crust * (1 - phi_s)   # Mass (kg/yr)
    m_p2o5_lost = m * p2o5_weight_fraction * P_alt * Lp2o5    # P2O5 released (kg/yr)
    m_p2o5_ocean = m_p2o5_lost * (1.0 - c_fixed)    # Delivered to ocean
    n_p2o5 = (m_p2o5_ocean * 1000) / w_mol_P2O5
    n_p = 2 * n_p2o5       # Convert to mol P

    return n_p

# ========== 5. Monte Carlo calculation ========== #

F_vals_left = compute_flux(
    L_vals_left, S_vals_left, D_vals_left,
    Hs_left, phi_s_left,
    P_alt_left, Lp2o5_left,
    c_fixed_left
)

F_vals_right = compute_flux(
    L_vals_right, S_vals_right, D_vals_right,
    Hs_right, phi_s_right,
    P_alt_right, Lp2o5_right,
    c_fixed_right
)

# ========== 6. Plotting ========== #

validA = np.isfinite(F_vals_left) & (F_vals_left > 0)
L_hex = L_vals_left[validA]
yA = np.log10(F_vals_left[validA])

validB = np.isfinite(F_vals_right) & (F_vals_right > 0)
c_hex = c_fixed_right[validB]
yB = np.log10(F_vals_right[validB])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

y_min = min(yA.min(), yB.min())
y_max = max(yA.max(), yB.max())

# -------- Left plot -------- #
hb1 = ax1.hexbin(L_hex, yA, gridsize=50, cmap='Blues', mincnt=1, bins='log')
ax1.set_xlabel('Subduction-zone length (m)')
ax1.set_ylabel('log(P flux, mol/yr)')
ax1.set_title('Flux vs Length (Random L)')
ax1.set_ylim(y_min, y_max)

river_flux = 2.8e10
river_flux_log = np.log10(river_flux)
ax1.axhline(river_flux_log, color='k', linestyle='--', linewidth=1.0)
ax1.text(5.0e4, river_flux_log + 0.05,
         'River flux:\n2.8×10¹⁰ mol/yr', fontsize=9)

cb1 = fig.colorbar(hb1, ax=ax1)
cb1.set_label('counts')

# -------- Right plot -------- #
hb2 = ax2.hexbin(c_hex, yB, gridsize=50, cmap='Blues', mincnt=1, bins='log')
ax2.set_xlabel('Phosphorus fixation efficiency (c_fixed)')
ax2.set_ylabel('log(P flux, mol/yr)')
ax2.set_title('Flux vs c_fixed (L = 5.0e6 m fixed)')
ax2.set_xlim(0, 1)
ax2.set_ylim(y_min, y_max)

cb2 = fig.colorbar(hb2, ax=ax2)
cb2.set_label('counts')

plt.tight_layout()
plt.show()
