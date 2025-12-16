import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Global parameters ==========
N = 1000000  # Number of Monte Carlo samples (large!)

rho_crust = 3300
w_mol_P2O5 = 141.94
p2o5_weight_fraction = 0.005  # 0.5 wt%

# ========== 2. Define parameter-sampling functions ==========

def sample_L(n):
    return np.random.uniform(0, 5.0e6, n)

def sample_L_fixed(n, L_value=5.0e6):
    return np.full(n, L_value)

def sample_S_triangular(n):
    return np.random.triangular(0.028, 0.09, 0.102, n)

def sample_D(n):
    return np.random.uniform(1000, 2000, n)

def sample_fraction(n, low, high):
    return np.random.uniform(low, high, n)


# ========== 3. Parameter assignment (Left vs Right) ==========

# A) Left: random L
L_vals_left = sample_L(N)
S_vals_left = sample_S_triangular(N)
D_vals_left = sample_D(N)

Hs_left    = sample_fraction(N, 0.5, 0.8)
Hd_left    = sample_fraction(N, 0.1, 0.2)
phi_s_left = sample_fraction(N, 0.12, 0.3)
phi_d_left = sample_fraction(N, 0.08, 0.12)
P_alt_left = sample_fraction(N, 0.6, 1.0)
Lp2o5_left = sample_fraction(N, 0.8, 0.9)

# ---- NEW: Combine cement + Fe into ONE parameter (0–1)
c_fixed_left = sample_fraction(N, 0.0, 1.0)

# B) Right: fixed L
L_vals_right = sample_L_fixed(N, L_value=5.0e6)
S_vals_right = sample_S_triangular(N)
D_vals_right = sample_D(N)

Hs_right    = sample_fraction(N, 0.5, 0.8)
Hd_right    = sample_fraction(N, 0.1, 0.2)
phi_s_right = sample_fraction(N, 0.12, 0.3)
phi_d_right = sample_fraction(N, 0.08, 0.12)
P_alt_right = sample_fraction(N, 0.6, 1.0)
Lp2o5_right = sample_fraction(N, 0.8, 0.9)

c_fixed_right = sample_fraction(N, 0.0, 1.0)


# ========== 4. Arrays to store results (Left vs Right) ==========
F_vals_left = np.zeros(N)
F_vals_right = np.zeros(N)


# ========== 5. Flux computation function (UPDATED) ==========

def compute_flux(L, S, D, Hs, Hd, phi_s, phi_d, P_alt, Lp2o5, c_fixed):
    """
    Compute P flux (mol/yr) for one sample.
    c_fixed is the total fraction fixed (cement + Fe), in [0,1].
    """

    # Shallow (0–1 km) vs deep
    t_s = min(D, 1000.0)
    t_d = max(D - 1000.0, 0.0)

    # Volume (m^3/yr)
    v_s = L * S * t_s * Hs
    v_d = L * S * t_d * Hd
    v_total = v_s + v_d

    # Mass (kg/yr)
    m_s = v_s * rho_crust * (1 - phi_s)
    m_d = v_d * rho_crust * (1 - phi_d)
    m_tot = m_s + m_d

    # Lost P2O5 (kg/yr)
    m_p2o5_lost = m_tot * p2o5_weight_fraction * P_alt * Lp2o5

    # P delivered to ocean
    m_p2o5_ocean = m_p2o5_lost * (1 - c_fixed)

    # mol/yr
    n_p2o5_ocean = (m_p2o5_ocean * 1000.0) / w_mol_P2O5
    return n_p2o5_ocean


# A) Left scenario
for i in range(N):
    F_vals_left[i] = compute_flux(
        L_vals_left[i], S_vals_left[i], D_vals_left[i],
        Hs_left[i], Hd_left[i], phi_s_left[i], phi_d_left[i],
        P_alt_left[i], Lp2o5_left[i],
        c_fixed_left[i]
    )

# B) Right scenario
for i in range(N):
    F_vals_right[i] = compute_flux(
        L_vals_right[i], S_vals_right[i], D_vals_right[i],
        Hs_right[i], Hd_right[i], phi_s_right[i], phi_d_right[i],
        P_alt_right[i], Lp2o5_right[i],
        c_fixed_right[i]
    )


# ========== 6. Plot two heatmaps ==========

validA = (F_vals_left > 0)
L_hex = L_vals_left[validA]
yA = np.log10(F_vals_left[validA])

validB = (F_vals_right > 0)
c_hex = c_fixed_right[validB]
yB = np.log10(F_vals_right[validB])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

y_min = min(yA.min(), yB.min())
y_max = max(yA.max(), yB.max())

# Left plot
hb1 = ax1.hexbin(L_hex, yA, gridsize=50, cmap='Blues', mincnt=1, bins='log')
ax1.set_xlabel('Subduction-zone length (linear)')
ax1.set_ylabel('log10(P flux (mol/yr))')
ax1.set_title('Flux vs Length (Random L)')
ax1.set_ylim(y_min, y_max)

river_flux = 2.8e10
river_flux_log = np.log10(river_flux)
ax1.axhline(river_flux_log, color='k', linestyle='--', linewidth=1.0)
ax1.text(5.0e4, river_flux_log + 0.05, 'River flux:\n2.8e10 mol/yr', fontsize=9)

cb1 = fig.colorbar(hb1, ax=ax1)
cb1.set_label('Data density (log(count))')

# Right plot
hb2 = ax2.hexbin(c_hex, yB, gridsize=50, cmap='Blues', mincnt=1, bins='log')
ax2.set_xlabel('Fixed fraction (c_fixed)')
ax2.set_ylabel('log10(P flux (mol/yr))')
ax2.set_title('Flux vs c_fixed (L=5.0e6 fixed)')
ax2.set_xlim(0, 1)
ax2.set_ylim(y_min, y_max)

cb2 = fig.colorbar(hb2, ax=ax2)
cb2.set_label('Data density (log(count))')

plt.tight_layout()
plt.savefig("final_model_updated.pdf", dpi=300, bbox_inches='tight')
plt.show()
