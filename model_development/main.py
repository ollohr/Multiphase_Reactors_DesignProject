"""
main.py
-------
Run the TFB Fischer-Tropsch reactor model, verify heat removal,
and produce all key plots for the design project report.

Figures produced
----------------
1. X_CO vs reactor volume V
2. Reaction rate r_V vs V
3. Effect of temperature on r_V  (sensitivity)
4. Effect of pressure on X_CO profile  (sensitivity)
5. Effect of H2/CO feed ratio on r_V  (sensitivity)
6. Reactor volume needed vs target conversion X_CO
7. Heat removal summary (bar chart: Q, A_cool, L_tube)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

from kinetics import (
    reaction_rate, volumetric_rate, arrhenius_coefficients,
    F_ACTIVITY, RU, T0
)
from ode_system import (
    solve_reactor, dF_dV, conversion,
    N_CO_IN, N_H2_IN, N_TOT_IN,
    T_BED, P_TOTAL, ETA, EPSILON, RHO_P, RHO_CAT_BULK, DHR,
    _partial_pressures
)

# ── Matplotlib style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 2.0,
})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — REACTOR SIZING
# ══════════════════════════════════════════════════════════════════════

def reactor_geometry(V_reactor: float) -> dict:
    """
    From total reactor volume, compute cross-section, diameter, height.
    Uses corrected operating superficial velocity u0 in turbulent regime.
    """
    # Volumetric gas flow at reactor inlet (ideal gas)
    R_gas = 8.314          # J/mol/K
    T_K   = T_BED
    P_Pa  = P_TOTAL * 1e5
    Vdot_gas = N_TOT_IN * R_gas * T_K / P_Pa   # m³/s

    # Operating superficial velocity — turbulent regime
    # u_c (onset turbulent) = 0.87 m/s from report
    # Use u0 = 1.5 × u_c = 1.30 m/s  (well inside turbulent regime)
    u0 = 1.30   # m/s

    A_cross  = Vdot_gas / u0                 # m²
    D        = np.sqrt(4 * A_cross / np.pi)  # m
    H_bed    = V_reactor / A_cross           # m (total bed height)

    return {
        "Vdot_gas": Vdot_gas,
        "u0":       u0,
        "A_cross":  A_cross,
        "D":        D,
        "H_bed":    H_bed,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — HEAT REMOVAL
# ══════════════════════════════════════════════════════════════════════

def heat_removal_check(X_CO: float = 0.80,
                       U: float = 400.0,
                       T_cool_C: float = 250.0,
                       d_o: float = 0.050) -> dict:
    """
    Compute and verify heat removal parameters.

    Parameters
    ----------
    X_CO    : target CO conversion
    U       : overall heat transfer coefficient [W/m²/K]
    T_cool_C: coolant temperature (°C)  — saturated water/steam
    d_o     : cooling tube outer diameter [m]
    """
    Q       = N_CO_IN * X_CO * abs(DHR)            # W
    dT_lm   = (T_BED - 273.15) - T_cool_C          # K (log-mean ≈ ΔT for isothermal bed)
    A_cool  = Q / (U * dT_lm)                      # m²
    L_tube  = A_cool / (np.pi * d_o)               # m

    # Saturation pressure of water (Antoine approximation, T in °C)
    # log10(P/bar) = 5.40221 - 1838.675/(T + 230.170)  [T in °C, P in bar]
    T_c = T_cool_C
    P_sat = 10 ** (5.40221 - 1838.675 / (T_c + 230.170))

    return {
        "Q_MW":     Q / 1e6,
        "dT_K":     dT_lm,
        "A_cool":   A_cool,
        "L_tube":   L_tube,
        "P_sat_bar": P_sat,
        "U":        U,
        "T_cool_C": T_cool_C,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — SENSITIVITY HELPERS
# ══════════════════════════════════════════════════════════════════════

def solve_with_params(T_C: float = 335.0,
                      P_bar: float = 20.0,
                      H2CO: float = 8.0,
                      X_target: float = 0.80,
                      V_max: float = 8000.0,
                      n_pts: int = 400) -> dict:
    """
    Solve ODE with custom T, P, H2/CO ratio.
    Returns same dict as solve_reactor().
    """
    T_K      = T_C + 273.15
    N_CO     = N_CO_IN
    N_H2     = N_CO * H2CO
    F0       = np.array([N_CO, N_H2, 0.0, 0.0])
    rho_bulk = RHO_P * (1.0 - EPSILON) * ETA

    def dF(V, F):
        F_tot = np.sum(F)
        if F_tot <= 0:
            return np.zeros(4)
        p_CO = F[0] / F_tot * P_bar
        p_H2 = F[1] / F_tot * P_bar
        rV   = volumetric_rate(p_CO, p_H2, T_K, rho_bulk)
        return np.array([-1, -2, +1, +1]) * rV

    def hit_target(V, F):
        return (N_CO - F[0]) / N_CO - X_target
    hit_target.terminal  = True
    hit_target.direction = +1

    sol = solve_ivp(dF, (0, V_max), F0, method="RK45",
                    events=hit_target, rtol=1e-6, atol=1e-8,
                    dense_output=True)

    V_end = sol.t_events[0][0] if sol.t_events[0].size > 0 else V_max
    V_arr = np.linspace(0, V_end, n_pts)
    Farr  = sol.sol(V_arr)
    X_arr = (N_CO - Farr[0]) / N_CO

    rV_arr = np.array([
        volumetric_rate(Farr[0, i] / max(Farr[:, i].sum(), 1e-9) * P_bar,
                        Farr[1, i] / max(Farr[:, i].sum(), 1e-9) * P_bar,
                        T_K, rho_bulk)
        for i in range(n_pts)
    ])

    return {"V": V_arr, "X_CO": X_arr, "rV": rV_arr, "V_reactor": V_end}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — PLOTTING
# ══════════════════════════════════════════════════════════════════════

def make_all_figures():

    # ── Base case solution ──────────────────────────────────────────
    base = solve_reactor(X_target=0.80, V_max=5000, n_points=500)
    geo  = reactor_geometry(base["V_reactor"])
    hr   = heat_removal_check(X_CO=0.80, U=400, T_cool_C=250)

    print("=" * 60)
    print("BASE CASE RESULTS  (T=335°C, P=20 bar, H2/CO=8, X=80%)")
    print("=" * 60)
    print(f"  V_reactor   = {base['V_reactor']:.1f}  m³")
    print(f"  D_reactor   = {geo['D']:.2f}  m")
    print(f"  H_bed       = {geo['H_bed']:.2f}  m")
    print(f"  u0 (oper.)  = {geo['u0']:.2f}  m/s  (turbulent regime)")
    print(f"  Q (heat)    = {hr['Q_MW']:.1f}  MW")
    print(f"  A_cool      = {hr['A_cool']:.0f}  m²")
    print(f"  L_tube      = {hr['L_tube']:.0f}  m")
    print(f"  ΔT          = {hr['dT_K']:.0f}  K  (bed - coolant)")
    print(f"  P_sat cool  = {hr['P_sat_bar']:.1f}  bar  @ {hr['T_cool_C']:.0f}°C")
    print()

    # ── Figure 1 & 2: X_CO and r_V vs V ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(base["V"], base["X_CO"] * 100, color=COLORS[0])
    ax.axvline(base["V_reactor"], color="gray", ls="--", lw=1.2,
               label=f"V = {base['V_reactor']:.0f} m³  (X = 80%)")
    ax.axhline(80, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Reactor volume V  [m³]")
    ax.set_ylabel("CO conversion X$_{CO}$  [%]")
    ax.set_title("CO conversion along reactor")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(base["V"], base["rV"], color=COLORS[1])
    ax.set_xlabel("Reactor volume V  [m³]")
    ax.set_ylabel("Volumetric rate r$_V$  [mol$_{CO}$ s$^{-1}$ m$^{-3}$]")
    ax.set_title("Reaction rate profile along reactor")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig1_conversion_rate.png",
                bbox_inches="tight")
    print("Saved fig1_conversion_rate.png")

    # ── Figure 3: Temperature sensitivity ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    temps = [310, 325, 335, 345, 360]
    ax = axes[0]
    for i, T_C in enumerate(temps):
        res = solve_with_params(T_C=T_C, V_max=5000)
        lbl = f"{T_C}°C  (V={res['V_reactor']:.0f} m³)"
        ax.plot(res["V"], res["X_CO"] * 100, color=COLORS[i % 6], label=lbl)
    ax.axhline(80, color="gray", ls=":", lw=1.0, label="80% target")
    ax.set_xlabel("Reactor volume V  [m³]")
    ax.set_ylabel("CO conversion  [%]")
    ax.set_title("Sensitivity: operating temperature")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    T_range = np.linspace(290, 380, 100) + 273.15
    r_inlet = []
    for T_K in T_range:
        p_CO = (1/9) * P_TOTAL
        p_H2 = (8/9) * P_TOTAL
        r_inlet.append(volumetric_rate(p_CO, p_H2, T_K, RHO_CAT_BULK))
    ax.plot(T_range - 273.15, r_inlet, color=COLORS[0])
    ax.axvline(335, color="red", ls="--", lw=1.2, label="Design T = 335°C")
    ax.set_xlabel("Temperature  [°C]")
    ax.set_ylabel("r$_V$ at inlet  [mol$_{CO}$ s$^{-1}$ m$^{-3}$]")
    ax.set_title("Inlet rate vs temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig2_temperature_sensitivity.png",
                bbox_inches="tight")
    print("Saved fig2_temperature_sensitivity.png")

    # ── Figure 4: Pressure sensitivity ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pressures = [10, 15, 20, 25, 30, 40]
    ax = axes[0]
    for i, P in enumerate(pressures):
        res = solve_with_params(P_bar=P, V_max=5000)
        lbl = f"{P} bar  (V={res['V_reactor']:.0f} m³)"
        ax.plot(res["V"], res["X_CO"] * 100, color=COLORS[i % 6], label=lbl)
    ax.axhline(80, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Reactor volume V  [m³]")
    ax.set_ylabel("CO conversion  [%]")
    ax.set_title("Sensitivity: operating pressure")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    P_range = np.linspace(5, 45, 100)
    r_p = []
    for P_b in P_range:
        p_CO = (1/9) * P_b
        p_H2 = (8/9) * P_b
        r_p.append(volumetric_rate(p_CO, p_H2, T_BED, RHO_CAT_BULK))
    ax.plot(P_range, r_p, color=COLORS[1])
    ax.axvline(P_TOTAL, color="red", ls="--", lw=1.2, label="Design P = 20 bar")
    ax.set_xlabel("Pressure  [bar]")
    ax.set_ylabel("r$_V$ at inlet  [mol$_{CO}$ s$^{-1}$ m$^{-3}$]")
    ax.set_title("Inlet rate vs pressure (T=335°C, H$_2$/CO=8)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig3_pressure_sensitivity.png",
                bbox_inches="tight")
    print("Saved fig3_pressure_sensitivity.png")

    # ── Figure 5: H2/CO ratio sensitivity ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios = [1, 2, 4, 6, 8, 10]
    for i, ratio in enumerate(ratios):
        res = solve_with_params(H2CO=ratio, V_max=5000)
        ax.plot(res["V"], res["X_CO"] * 100, color=COLORS[i % 6],
                label=f"H$_2$/CO = {ratio}  (V={res['V_reactor']:.0f} m³)")
    ax.axhline(80, color="gray", ls=":", lw=1.0, label="80% target")
    ax.axvline(base["V_reactor"], color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Reactor volume V  [m³]")
    ax.set_ylabel("CO conversion  [%]")
    ax.set_title("Sensitivity: H₂/CO feed ratio  (T=335°C, P=20 bar)")
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(left=0); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig4_h2co_sensitivity.png",
                bbox_inches="tight")
    print("Saved fig4_h2co_sensitivity.png")

    # ── Figure 6: V_reactor vs target X ────────────────────────────
    X_targets  = np.linspace(0.30, 0.95, 30)
    V_required = []
    for Xt in X_targets:
        res = solve_with_params(X_target=float(Xt), V_max=8000)
        V_required.append(res["V_reactor"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_targets * 100, V_required, color=COLORS[0], marker="o",
            markersize=4)
    ax.axvline(80, color="red", ls="--", lw=1.2,
               label=f"Design point: 80%  →  {base['V_reactor']:.0f} m³")
    ax.set_xlabel("Target CO conversion X$_{CO}$  [%]")
    ax.set_ylabel("Required reactor volume V  [m³]")
    ax.set_title("Reactor volume vs target conversion  (T=335°C, P=20 bar)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 95)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig5_volume_vs_conversion.png",
                bbox_inches="tight")
    print("Saved fig5_volume_vs_conversion.png")

    # ── Figure 7: Heat removal summary ─────────────────────────────
    T_cool_options = [200, 220, 250]
    labels   = [f"T$_{{cool}}$={T}°C" for T in T_cool_options]
    A_vals   = []
    L_vals   = []
    Psat_vals = []
    for Tc in T_cool_options:
        h = heat_removal_check(T_cool_C=Tc)
        A_vals.append(h["A_cool"])
        L_vals.append(h["L_tube"] / 1000)   # km
        Psat_vals.append(h["P_sat_bar"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(labels))
    w = 0.5
    c = [COLORS[2], COLORS[0], COLORS[1]]

    ax = axes[0]
    bars = ax.bar(x, A_vals, width=w, color=c)
    ax.bar_label(bars, fmt="%.0f m²", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Cooling area A$_{cool}$  [m²]")
    ax.set_title("Required cooling area")
    ax.set_ylim(0, max(A_vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, L_vals, width=w, color=c)
    ax.bar_label(bars, fmt="%.1f km", padding=4, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Total tube length  [km]")
    ax.set_title("Required tube length  (d$_o$=50 mm)")
    ax.set_ylim(0, max(L_vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    bars = ax.bar(x, Psat_vals, width=w, color=c)
    ax.bar_label(bars, fmt="%.1f bar", padding=4, fontsize=9)
    ax.axhline(P_TOTAL, color="red", ls="--", lw=1.2,
               label="Reactor P = 20 bar")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Coolant saturation pressure  [bar]")
    ax.set_title("Steam pressure in cooling loop")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(Psat_vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Heat removal comparison  (Q = {hr['Q_MW']:.1f} MW, U = 400 W/m²/K)",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/fig6_heat_removal.png",
                bbox_inches="tight")
    print("Saved fig6_heat_removal.png")

    plt.close("all")
    print("\nAll figures saved to /mnt/user-data/outputs/")


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_all_figures()


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — FINAL REACTOR SUMMARY (run directly)
# ══════════════════════════════════════════════════════════════════════

def print_design_summary():
    """Print a complete, self-consistent reactor design summary."""
    base = solve_reactor(X_target=0.80, V_max=5000)
    geo  = reactor_geometry(base["V_reactor"])
    hr   = heat_removal_check(X_CO=0.80, U=400, T_cool_C=250)

    # Bed height is controlled by cooling requirement (pitch = 100 mm)
    pitch    = 0.10   # m
    pf       = 0.70   # packing factor for circular vessel
    n_tubes  = int(pf * geo["A_cross"] / pitch**2)
    A_per_m  = n_tubes * np.pi * 0.05
    H_cool   = hr["A_cool"] / A_per_m      # m — minimum H from cooling

    # Final bed height = max of kinetic requirement and cooling requirement
    H_final = max(geo["H_bed"], H_cool)
    V_final = geo["A_cross"] * H_final

    print("\n" + "="*60)
    print("FINAL INTEGRATED REACTOR DESIGN SUMMARY")
    print("="*60)
    print(f"  Operating T          = 335°C  (608 K)")
    print(f"  Operating P          = 20 bar")
    print(f"  H2/CO feed ratio     = 8")
    print(f"  CO conversion target = 80%")
    print()
    print(f"  --- Kinetics ---")
    print(f"  V_reactor (kinetics) = {base['V_reactor']:.1f} m³")
    print(f"  Note: V is small because of high F_activity=3 and")
    print(f"        ρ_p=1800 kg/m³ (porous iron carbide composite)")
    print()
    print(f"  --- Geometry ---")
    print(f"  D_reactor            = {geo['D']:.2f} m")
    print(f"  A_cross              = {geo['A_cross']:.2f} m²")
    print(f"  u₀ (operating)       = {geo['u0']:.2f} m/s  (turbulent regime)")
    print(f"  H_bed (kinetics)     = {geo['H_bed']:.2f} m")
    print(f"  H_bed (cooling req.) = {H_cool:.2f} m  ← controls design")
    print(f"  H_bed (final)        = {H_final:.2f} m")
    print(f"  V_total (final)      = {V_final:.0f} m³")
    print()
    print(f"  --- Heat removal ---")
    print(f"  Q                    = {hr['Q_MW']:.1f} MW")
    print(f"  U                    = {hr['U']:.0f} W/m²/K")
    print(f"  T_cool               = {hr['T_cool_C']:.0f}°C  (P_sat = {hr['P_sat_bar']:.1f} bar)")
    print(f"  ΔT (bed - coolant)   = {hr['dT_K']:.0f} K")
    print(f"  A_cool               = {hr['A_cool']:.0f} m²")
    print(f"  n_tubes (pitch=100mm)= {n_tubes}")
    print(f"  H_bed from cooling   = {H_cool:.2f} m  (single-pass straight tubes)")
    print(f"  Tube OD              = 50 mm")
    print()
    print(f"  [SAS reference: D=8-10.7 m, 340°C, 20-40 bar  — Steynberg 1999]")


if __name__ == "__main__":
    make_all_figures()
    print_design_summary()
