import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 8.3145  # J/(mol K)

# Reactor conditions
T0 = 340 + 273.15           # K
p_bar = 20 * 1.01325        # bar

# Kinetic parameters
a0 = 8.88522e-3             # mol/(s kg_cat bar^2)
Ea = 3.737e4                # J/mol
b0 = 2.226                  # 1/bar
DbH = -6.837e3              # J/mol

# Catalyst parameters
cat_F = 3.0                 # catalyst activity factor (Appendix A)
n_eff = 0.9                 # effectiveness factor

# Fixed inlet CO flow (from 3000 ton CO/day requirement)
F_CO0 = 1239.6              # mol/s

# Temperature-dependent kinetics at T0
a = a0 * np.exp(Ea / R * (1/493.15 - 1/T0))
b = b0 * np.exp(DbH / R * (1/493.15 - 1/T0))

# --- Sweep H2/CO feed ratio ---
ratios = np.linspace(0.2, 12, 500)            # H2/CO molar feed ratio [-]

r_intrinsic  = np.zeros_like(ratios)          # mol/(s kg_cat)       -- intrinsic rate at inlet
r_per_feed   = np.zeros_like(ratios)          # mol CO / (s kg_cat) per (mol/s total feed)

for i, ratio in enumerate(ratios):
    F_H2_in = ratio * F_CO0                   # mol/s
    F_tot   = F_CO0 + F_H2_in                 # mol/s  (no HC or H2O at inlet)

    y_CO = F_CO0 / F_tot
    y_H2 = F_H2_in / F_tot

    p_CO = y_CO * p_bar                       # bar
    p_H2 = y_H2 * p_bar                       # bar

    # Yates & Satterfield intrinsic rate
    r_int = cat_F * a * p_CO * p_H2 / (1 + b * p_CO)**2   # mol/(s kg_cat)
    r_eff = n_eff * r_int                                   # mol/(s kg_cat)

    r_intrinsic[i] = r_eff

    # Normalise by total molar feed: productivity per unit feed flowrate
    # Units: mol_CO_reacted / (s kg_cat) / (mol_total / s) = mol_CO / mol_feed / kg_cat
    r_per_feed[i]  = r_eff / F_tot

# --- Find optimum ---
idx_opt    = np.argmax(r_per_feed)
ratio_opt  = ratios[idx_opt]
r_opt      = r_per_feed[idx_opt]

# Reference points for annotation
ref_ratios = [2.0, 2.5, 3.0, ratio_opt, 8.0]
ref_labels = ["2:1\n(stoich. long chain)", "2.5:1",
              "3:1\n(stoich. CH₄)", f"{ratio_opt:.2f}:1\n(optimum)", "8:1"]
ref_colors = ["steelblue", "seagreen", "darkorange", "crimson", "grey"]

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: absolute intrinsic rate at reactor inlet
ax = axes[0]
ax.plot(ratios, r_intrinsic, color="steelblue", linewidth=2)
for rv, rl, rc in zip(ref_ratios, ref_labels, ref_colors):
    idx = np.argmin(np.abs(ratios - rv))
    ax.axvline(rv, color=rc, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.plot(rv, r_intrinsic[idx], "o", color=rc, markersize=6)
ax.set_xlabel(r"H$_2$/CO molar feed ratio [-]")
ax.set_ylabel(r"Effective reaction rate $r_{\rm eff}$ [mol s$^{-1}$ kg$_{\rm cat}^{-1}$]")
ax.set_title("Inlet reaction rate vs. H₂/CO ratio")
ax.set_xlim(0, 12)
ax.grid(True)

# Right: rate normalised by total feed (true productivity measure)
ax = axes[1]
ax.plot(ratios, r_per_feed * 1e3, color="steelblue", linewidth=2)
for rv, rl, rc in zip(ref_ratios, ref_labels, ref_colors):
    idx = np.argmin(np.abs(ratios - rv))
    ax.axvline(rv, color=rc, linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"H₂/CO = {rl}")
    ax.plot(rv, r_per_feed[idx] * 1e3, "o", color=rc, markersize=6)
ax.axvline(ratio_opt, color="crimson", linestyle="-", linewidth=1.8)
ax.set_xlabel(r"H$_2$/CO molar feed ratio [-]")
ax.set_ylabel(r"$r_{\rm eff}$ / $F_{\rm total}$ [mmol mol$^{-1}$ kg$_{\rm cat}^{-1}$]")
ax.set_title("CO productivity per unit feed vs. H₂/CO ratio")
ax.set_xlim(0, 12)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True)

print(f"Optimal H2/CO feed ratio (max productivity per mol feed): {ratio_opt:.2f}")
print(f"  r_eff at optimum : {r_intrinsic[idx_opt]:.4f} mol/(s kg_cat)")
print(f"  r/F_tot at optimum: {r_opt*1e3:.4f} mmol/mol/kg_cat")
print(f"  r/F_tot at ratio=2 : {r_per_feed[np.argmin(np.abs(ratios-2))]*1e3:.4f} mmol/mol/kg_cat")
print(f"  r/F_tot at ratio=8 : {r_per_feed[np.argmin(np.abs(ratios-8))]*1e3:.4f} mmol/mol/kg_cat")

plt.tight_layout()
plt.show()