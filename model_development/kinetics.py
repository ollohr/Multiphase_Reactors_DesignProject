"""
kinetics.py
-----------
Yates & Satterfield (1991) Fischer-Tropsch kinetics as given in
the project Appendix A (Maretto & Krishna parameterisation).

    r = F * a(T) * p_CO * p_H2 / (1 + b(T) * p_CO)^2

Units:
    r    [mol_CO / s / kg_cat]
    p_i  [bar]
    T    [K]
"""

import numpy as np

# ── Kinetic constants (Table A.1 in project brief) ──────────────────
A0  = 8.88522e-3   # mol/s/kg_cat/bar²  — rate coeff at 493.15 K
EA  = 3.737e4      # J/mol              — activation energy
B0  = 2.226        # bar⁻¹             — adsorption coeff at 493.15 K
DBH = -6.837e3     # J/mol             — adsorption enthalpy
T0  = 493.15       # K                 — reference temperature
RU  = 8.314        # J/mol/K           — universal gas constant
F_ACTIVITY = 3.0   # catalyst activity multiplication factor (Appendix A)


def arrhenius_coefficients(T: float) -> tuple[float, float]:
    """
    Return temperature-dependent rate coefficient a [mol/s/kg_cat/bar²]
    and adsorption coefficient b [bar⁻¹] at temperature T [K].
    """
    a = A0 * np.exp((EA  / RU) * (1.0 / T0 - 1.0 / T))
    b = B0 * np.exp((DBH / RU) * (1.0 / T0 - 1.0 / T))
    return a, b


def reaction_rate(p_CO: float, p_H2: float, T: float) -> float:
    """
    Intrinsic reaction rate per unit catalyst mass.

    Parameters
    ----------
    p_CO : float   Partial pressure of CO  [bar]
    p_H2 : float   Partial pressure of H2  [bar]
    T    : float   Temperature             [K]

    Returns
    -------
    r : float   [mol_CO / s / kg_cat]   (≥ 0)
    """
    a, b = arrhenius_coefficients(T)
    r = F_ACTIVITY * a * p_CO * p_H2 / (1.0 + b * p_CO) ** 2
    return max(r, 0.0)


def volumetric_rate(p_CO: float, p_H2: float, T: float,
                    rho_cat_bulk: float) -> float:
    """
    Reaction rate per unit reactor volume.

        r_V = r * rho_cat_bulk   [mol_CO / s / m³_reactor]

    Parameters
    ----------
    rho_cat_bulk : float   Bulk catalyst density in bed [kg_cat / m³_reactor]
                           = rho_particle * (1 - epsilon) * eta
    """
    return reaction_rate(p_CO, p_H2, T) * rho_cat_bulk


if __name__ == "__main__":
    # Quick sanity check at design conditions
    T_op = 335 + 273.15      # K
    P    = 20.0              # bar total
    # Feed: CO:H2 = 1:8 (molar)
    y_CO = 1.0 / 9.0
    y_H2 = 8.0 / 9.0
    p_CO = y_CO * P
    p_H2 = y_H2 * P

    a, b = arrhenius_coefficients(T_op)
    r    = reaction_rate(p_CO, p_H2, T_op)

    print("=== Kinetics sanity check @ 335°C, 20 bar, CO:H2=1:8 ===")
    print(f"  a(T)  = {a:.4e} mol/s/kg_cat/bar²")
    print(f"  b(T)  = {b:.4f} bar⁻¹")
    print(f"  p_CO  = {p_CO:.3f} bar")
    print(f"  p_H2  = {p_H2:.3f} bar")
    print(f"  r     = {r:.4e} mol_CO/s/kg_cat")
