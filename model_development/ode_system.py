"""
ode_system.py
-------------
1-D pseudo-homogeneous plug-flow ODE model for the
Turbulent Fluidized Bed (TFB) Fischer-Tropsch reactor.

Reaction (simplified, n=1):
    CO  +  2 H2  →  -CH2-  +  H2O      ΔHr = -170 kJ/mol_CO

State vector  y = [F_CO, F_H2, F_HC, F_H2O]   [mol/s]
Independent variable: reactor volume V [m³]

Assumptions
-----------
1. Gas phase: plug flow (1-D axial)
2. Solid phase: perfectly back-mixed (isothermal bed)
3. Steady state
4. Isothermal (justified by intense solid mixing in TFB)
5. No internal or external mass-transfer limitation (η = 0.9 included
   in rho_cat_bulk, particles are 100 µm)
6. Ideal gas → partial pressures from mole fractions
7. Total molar flow changes with reaction (Δn = -2 per mol CO reacted)
"""

import numpy as np
from scipy.integrate import solve_ivp
from kinetics import volumetric_rate


# ── Fixed operating conditions ───────────────────────────────────────
T_BED   = 335.0 + 273.15   # K   reactor temperature (isothermal)
P_TOTAL = 20.0              # bar total pressure
ETA     = 0.90              # effectiveness factor (100 µm particles)
EPSILON = 0.55              # bed voidage in turbulent regime
RHO_P   = 1800.0   # kg/m³  effective particle density (porous iron carbide/carbon composite)

# Bulk catalyst density in the bed [kg_cat / m³_reactor]
RHO_CAT_BULK = RHO_P * (1.0 - EPSILON) * ETA   # ≈ 3155 kg/m³

# Feed specification
N_CO_IN  = 1239.6           # mol/s   (3000 t CO/day)
H2_CO_FEED_RATIO = 8.0      # H2:CO molar feed ratio
N_H2_IN  = N_CO_IN * H2_CO_FEED_RATIO
N_TOT_IN = N_CO_IN + N_H2_IN

DHR = -170_000.0            # J/mol_CO   reaction enthalpy

# ── Stoichiometry (per mol CO reacted) ──────────────────────────────
#   CO  +  2H2  →  CH2  +  H2O
#   ν_CO=-1, ν_H2=-2, ν_HC=+1, ν_H2O=+1   → Δn_mol = -1 per mol CO
NU = np.array([-1.0, -2.0, +1.0, +1.0])   # CO, H2, HC, H2O


def _partial_pressures(F: np.ndarray) -> tuple[float, float]:
    """
    Return (p_CO, p_H2) in bar given molar flow vector F [mol/s].
    F = [F_CO, F_H2, F_HC, F_H2O]
    """
    F_tot = np.sum(F)
    if F_tot <= 0:
        return 0.0, 0.0
    y_CO = F[0] / F_tot
    y_H2 = F[1] / F_tot
    return y_CO * P_TOTAL, y_H2 * P_TOTAL


def dF_dV(V: float, F: np.ndarray) -> np.ndarray:
    """
    ODE right-hand side:  dF/dV = ν * r_V

    Parameters
    ----------
    V : float        reactor volume coordinate [m³]
    F : array (4,)   molar flows [mol/s]  — CO, H2, HC, H2O

    Returns
    -------
    dF : array (4,)  [mol/s / m³]
    """
    p_CO, p_H2 = _partial_pressures(F)
    r_V = volumetric_rate(p_CO, p_H2, T_BED, RHO_CAT_BULK)
    return NU * r_V


def conversion(F_CO: float) -> float:
    """X_CO from current F_CO."""
    return (N_CO_IN - F_CO) / N_CO_IN


def solve_reactor(X_target: float = 0.80,
                  V_max: float = 5000.0,
                  n_points: int = 500) -> dict:
    """
    Integrate the ODE from V=0 until X_CO reaches X_target
    (or V_max is hit).

    Returns a dict with V, F arrays and derived quantities.
    """
    F0 = np.array([N_CO_IN, N_H2_IN, 0.0, 0.0])  # inlet

    # Event: stop when X_CO reaches X_target
    def target_conversion(V, F):
        return conversion(F[0]) - X_target
    target_conversion.terminal  = True
    target_conversion.direction = +1

    V_span = (0.0, V_max)
    V_eval = np.linspace(0.0, V_max, n_points)

    sol = solve_ivp(
        dF_dV,
        V_span,
        F0,
        method="RK45",
        t_eval=V_eval,
        events=target_conversion,
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
    )

    # Trim to the event point (or full range if not reached)
    if sol.t_events[0].size > 0:
        V_end = sol.t_events[0][0]
    else:
        V_end = V_max

    V_fine = np.linspace(0.0, V_end, n_points)
    F_fine = sol.sol(V_fine)              # shape (4, n_points)

    X_profile = (N_CO_IN - F_fine[0]) / N_CO_IN

    # Reaction rate profile
    rV_profile = np.array([
        volumetric_rate(*_partial_pressures(F_fine[:, i]),
                        T_BED, RHO_CAT_BULK)
        for i in range(n_points)
    ])

    return {
        "V":         V_fine,              # m³
        "F_CO":      F_fine[0],           # mol/s
        "F_H2":      F_fine[1],
        "F_HC":      F_fine[2],
        "F_H2O":     F_fine[3],
        "X_CO":      X_profile,           # –
        "rV":        rV_profile,          # mol/s/m³
        "V_reactor": V_end,               # m³  total volume for X_target
    }


if __name__ == "__main__":
    res = solve_reactor(X_target=0.80)
    print(f"V_reactor for 80% conversion = {res['V_reactor']:.1f} m³")
    print(f"Final X_CO                   = {res['X_CO'][-1]:.4f}")
    print(f"F_CO out                     = {res['F_CO'][-1]:.1f} mol/s")
