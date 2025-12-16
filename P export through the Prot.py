import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ==============================================================
# 1. Load and preprocess FORCE tectonic data
# ==============================================================

data = pd.read_excel('C:/Users/86152/Desktop/force.xlsx')
subset = data[(data['Time (Ma)'] >= 540) & (data['Time (Ma)'] <= 1799)].copy()
time = subset['Time (Ma)'].values.astype(np.float64)

ridge_length = subset['Mid-ocean ridge length (km)'].values
spreading_rate = subset['Spreading rate mean (cm/yr)'].values
crustal_prod_area = subset['Crustal production (km2/yr)'].values     # MOR source
subduction_length = subset['Subduction zone length (km)'].values     # Arc source
convergence_rate = subset['Convergence velocity mean (cm/yr)'].values
crustal_destr_area = subset['Crustal destruction  rates (km2/yr)'].values  # Arc source
cont_arc_length = subset['Continental arc length (mean) (km)'].values

# ==============================================================
# 2. Monte Carlo parameter settings
# ==============================================================

N = 10000  # Monte Carlo samples per timestep (adjustable)

density = 3300  # kg/m³ (consistent with first Monte Carlo model)
basalt_thickness = 7.0  # km
epsilon = 1e-10  # avoids log10 underflow

# Parameter distributions
P2O5_loss_frac_dist = (0.002, 0.006)
palagonite_frac_dist = (0.6, 1.0)
alt_depth_dist = (1.0, 2.0)   # km, equivalent to 1000–2000 m
hyalo_frac_dist = (0.2, 0.8)

# MOR vs Subduction-zone alteration parameters
Hs_dist    = (0.5, 0.8)    # subduction-zone effective alteration fraction
Hd_dist    = (0.1, 0.2)    # MOR effective alteration fraction
phi_s_dist = (0.12, 0.3)   # subduction-zone porosity
phi_d_dist = (0.08, 0.12)  # MOR porosity

# Fraction of dissolved P fixed near the water–rock interface
c_fixed_value = 0.9

# ==============================================================
# 3. Storage arrays
# ==============================================================

def init_array():
    return np.zeros_like(time, dtype=np.float64)

F_P_total = init_array()
F_P_total_low = init_array()
F_P_total_high = init_array()

P_total_all = np.zeros((len(time), N), dtype=np.float64)

# ==============================================================
# 4. Monte Carlo simulation
# ==============================================================

for i in range(len(time)):

    # ---- Random sampling of alteration parameters ----
    alt_depth_samples = np.random.uniform(*alt_depth_dist, N)
    hyalo_frac_samples = np.random.uniform(*hyalo_frac_dist, N)
    palag_frac_samples = np.random.uniform(*palagonite_frac_dist, N)
    P_loss_samples = np.random.uniform(*P2O5_loss_frac_dist, N)

    # MOR vs Subduction parameters
    Hs_samples    = np.random.uniform(*Hs_dist, N)
    Hd_samples    = np.random.uniform(*Hd_dist, N)
    phi_s_samples = np.random.uniform(*phi_s_dist, N)
    phi_d_samples = np.random.uniform(*phi_d_dist, N)

    # P retention fraction
    fix_frac_samples = np.full(N, c_fixed_value)

    # ---- MOR basalt volume (km³/yr) ----
    vol_basalt_MOR = crustal_prod_area[i] * alt_depth_samples * hyalo_frac_samples
    vol_basalt_MOR_eff = vol_basalt_MOR * Hd_samples

    # ---- Subduction/arc basalt volume ----
    if subduction_length[i] > 0:
        frac_oceanic_arc = max(0, (subduction_length[i] - cont_arc_length[i]) / subduction_length[i])
    else:
        frac_oceanic_arc = 0

    vol_basalt_arc = 0.2 * crustal_destr_area[i] * basalt_thickness * frac_oceanic_arc
    vol_basalt_arc *= hyalo_frac_samples
    vol_basalt_arc_eff = vol_basalt_arc * Hs_samples

    # ---- Convert volume → mass (kg/yr) ----
    mass_basalt_MOR = vol_basalt_MOR_eff * (1 - phi_d_samples) * 1e9 * density
    mass_basalt_arc = vol_basalt_arc_eff * (1 - phi_s_samples) * 1e9 * density

    # ---- P2O5 mass ----
    P2O5_mass_MOR = mass_basalt_MOR * palag_frac_samples * P_loss_samples
    P2O5_mass_arc = mass_basalt_arc * palag_frac_samples * P_loss_samples

    # ---- Convert P2O5 → P (mol/yr) ----
    moles_P_MOR = P2O5_mass_MOR * 1000.0 / 141.94 * 2.0
    moles_P_arc = P2O5_mass_arc * 1000.0 / 141.94 * 2.0

    # ---- P released to seawater ----
    moles_P_total = (moles_P_MOR + moles_P_arc) * (1 - fix_frac_samples)

    def safe_log10(x):
        return np.log10(np.maximum(x, epsilon))

    F_P_total[i] = safe_log10(np.median(moles_P_total))
    F_P_total_low[i] = safe_log10(np.percentile(moles_P_total, 2.5))
    F_P_total_high[i] = safe_log10(np.percentile(moles_P_total, 97.5))
    P_total_all[i, :] = safe_log10(moles_P_total)

# ==============================================================
# 5. Plot density heatmap
# ==============================================================

logP_bins = np.linspace(8, 12, 200)
Z = np.zeros((len(logP_bins) - 1, len(time)))

for i in range(len(time)):
    hist, _ = np.histogram(P_total_all[i, :], bins=logP_bins, density=True)
    Z[:, i] = hist

fig, ax = plt.subplots(figsize=(9, 6))
X, Y = np.meshgrid(time, logP_bins[:-1])

c = ax.pcolormesh(
    X, Y, Z,
    shading='auto',
    cmap='Blues',
    norm=Normalize(vmin=0, vmax=np.max(Z))
)

ax.plot(time, F_P_total, color='crimson', lw=2, label='Median P flux')
ax.axhline(np.log10(2.8e10), color='gray', linestyle='--', linewidth=1)
ax.text(time.max(), np.log10(2.8e10) + 0.1,
        'Modern river flux',
        color='gray', fontsize=9, ha='right', va='bottom')

ax.invert_xaxis()
ax.set_xlabel('Time (Ma)')
ax.set_ylabel('log10(P flux) [mol/yr]')
ax.set_title('Density plot of simulated submarine total P flux (1800–540 Ma)')
fig.colorbar(c, ax=ax, label='Probability density')

plt.tight_layout()
plt.savefig("C:/Users/86152/Desktop/model_density_heatmap.pdf", dpi=300)
plt.show()
