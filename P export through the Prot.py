import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

# ========== 1. Load data ========== #

data    = pd.read_excel('C:/Users/86152/Desktop/force.xlsx')   # Your file location
subset  = data[(data['Time (Ma)'] >= 540) & (data['Time (Ma)'] <= 1799)]

time       = subset['Time (Ma)'].values.astype(float)
crust_prod = subset['Crustal production (km2/yr)'].values
crust_dest = subset['Crustal destruction  rates (km2/yr)'].values
sub_len    = subset['Subduction zone length (km)'].values
arc_len    = subset['Continental arc length (mean) (km)'].values

# ========== 2. Fixed parameters ========== #

N                = 100000
rho_crust        = 3300.0        # kg/m³
basalt_thickness = 7.0           # km
w_mol_P2O5       = 141.94        # g/mol
p2o5_weight_fraction = 0.005     # 0.5 wt% P2O5 in fresh basalt
c_fixed          = 0.9           # Fraction of P fixed in seafloor
eps              = 1e-10

# ========== 3. Stochastic parameter ranges ========== #

params = {
    "Lp2o5":  (0.002, 0.006),   # P2O5 leaching fraction during alteration
    "P_alt":  (0.6,   1.0),     # Fraction of basalt undergoing alteration
    "depth":  (1.0,   2.0),     # Alteration depth (km)
    "hyalo":  (0.2,   0.8),     # Hyaloclastite fraction
    "Hs":     (0.5,   0.8),     # Hydration degree – subduction
    "Hd":     (0.1,   0.2),     # Hydration degree – MOR
    "phi_s":  (0.12,  0.3),     # Porosity – subduction basalt
    "phi_d":  (0.08,  0.12),    # Porosity – MOR basalt
}

HDI_probs = {"10": 0.10, "30": 0.30, "50": 0.50, "70": 0.70, "90": 0.90}

# ========== 4. Storage arrays ========== #

n_t     = len(time)
median  = np.zeros(n_t)
HDI_low  = {k: np.zeros(n_t) for k in HDI_probs}
HDI_high = {k: np.zeros(n_t) for k in HDI_probs}

# ========== 5. Functions ========== #

def compute_hdi(samples, prob):
    """Return the shortest interval covering `prob` of the sorted samples."""
    x      = np.sort(samples)
    n      = len(x)
    window = int(np.floor(prob * n))
    widths = x[window:] - x[:n - window]
    idx    = np.argmin(widths)
    return x[idx], x[idx + window]

def safe_log10(x):
    return np.log10(np.maximum(x, eps))

# ========== 6. Monte Carlo ========== #

rng = np.random.default_rng(42)

for i in range(n_t):

    samp = {k: rng.uniform(*v, N) for k, v in params.items()}

    # -- MOR contribution --
    vol_MOR  = crust_prod[i] * samp["depth"] * samp["hyalo"] * samp["Hd"]
    mass_MOR = vol_MOR * (1 - samp["phi_d"]) * 1e9 * rho_crust   # kg/yr

    # -- Subduction-arc contribution --
    frac     = max(0, (sub_len[i] - arc_len[i]) / sub_len[i]) if sub_len[i] > 0 else 0
    vol_arc  = 0.2 * crust_dest[i] * basalt_thickness * frac
    vol_arc *= samp["hyalo"] * samp["Hs"]
    mass_arc = vol_arc * (1 - samp["phi_s"]) * 1e9 * rho_crust   # kg/yr

    # -- P flux (mol/yr) --
    P_MOR = mass_MOR * samp["P_alt"] * samp["Lp2o5"]
    P_arc = mass_arc * samp["P_alt"] * samp["Lp2o5"]

    mol  = (P_MOR + P_arc) * 1000 / w_mol_P2O5 * 2   # P2O5 → mol P
    mol *= (1 - c_fixed)

    logP = safe_log10(mol)

    median[i] = np.median(logP)
    for k, p in HDI_probs.items():
        HDI_low[k][i], HDI_high[k][i] = compute_hdi(logP, p)

# ========== 7. Plot ========== #

HDI_colors = {
    "90": "#c6dbef",
    "70": "#9ecae1",
    "50": "#6baed6",
    "30": "#3182bd",
    "10": "#08519c",
}

fig, ax = plt.subplots(figsize=(10, 6))

for level in ["90", "70", "50", "30", "10"]:
    ax.fill_between(time, HDI_low[level], HDI_high[level],
                    color=HDI_colors[level], linewidth=0)

ax.plot(time, median, color='crimson', lw=2.0, label='Median P flux')

# -- Modern river flux reference line --
modern_flux_log = np.log10(2.8e10)
ax.axhline(modern_flux_log, color='gray', linestyle='--', linewidth=1.0)
ax.text(time.max(), modern_flux_log + 0.08,
        'Modern river flux', color='gray', fontsize=9, ha='right')

ax.invert_xaxis()
ax.set_xlabel('Time (Ma)')
ax.set_ylabel('log$_{10}$(P flux)  [mol yr$^{-1}$]')
ax.set_title('Highest-density probability intervals (HDI) of submarine P flux')
ax.legend(frameon=False, loc='upper left')

# -- Colorbar --
cmap = ListedColormap(['white'] + [HDI_colors[k] for k in ["90", "70", "50", "30", "10"]])
norm = BoundaryNorm(np.arange(-0.5, 6), cmap.N)
sm   = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=range(6))
cbar.ax.set_yticklabels(['Outside', '90% HDI', '70% HDI', '50% HDI', '30% HDI', '10% HDI'])

plt.tight_layout()
plt.show()
