import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
R = 8.3145  # J/(mol K)

# Reactor conditions
T0 = 340 + 273.15           # K
p_bar = 20 * 1.01325        # bar
p_pa = p_bar * 1e5          # Pa
V_fin = 431.3               # m^3

# Bed properties
epsilon_c = 0.55
rho_s = 7.794e3             # kg/m^3
rho_cat_bed = (1 - epsilon_c) * rho_s

cat_F = 3.0
n_eff = 0.9

# Kinetic parameters
a0 = 8.88522e-3             # mol/(s kg_cat bar^2)
Ea = 3.737e4                # J/mol
b0 = 2.226                  # 1/bar
DbH = -6.837e3              # J/mol

# Reaction enthalpy (per mol CO reacted)
dHrxn = -170e3              # J/mol CO reacted

# Approximate constant heat capacities [J/(mol K)]
Cp_CO = 30.0
Cp_H2 = 29.0
Cp_HC = 60.0
Cp_H2O = 40.0
Cp = np.array([Cp_CO, Cp_H2, Cp_HC, Cp_H2O])

# Inlet molar flows [mol/s]
F_CO0 = 1239.6
F_H20 = 2.5 * F_CO0
F_HC0 = 0.0
F_H2O0 = 0.0

F0 = np.array([F_CO0, F_H20, F_HC0, F_H2O0])

# Stoichiometry: CO + 2H2 -> HC + H2O
stoi_mat = np.array([-1.0, -2.0, 1.0, 1.0])


def mixed_temperature(F, T0, Cp, dHrxn, F_CO0):
    """
    Compute the perfectly mixed adiabatic temperature based on
    cumulative reaction up to the current reactor volume.
    """
    F_CO = F[0]

    # Total molar heat capacity flow [J/(s K)]
    Cp_flow = np.dot(F, Cp)

    # Prevent divide-by-zero
    Cp_flow = max(Cp_flow, 1e-12)

    # Moles of CO reacted up to current volume [mol/s equivalent basis]
    reacted_CO = max(F_CO0 - F_CO, 0.0)

    # Adiabatic mixed temperature
    T_mix = T0 + ((-dHrxn) * reacted_CO) / Cp_flow

    return T_mix


def hybrid_pfr_mass_balance(V, F, params):
    """
    PFR mass balances with a perfectly mixed adiabatic temperature.
    Temperature is computed algebraically from cumulative conversion.
    """
    (
        a0, b0, Ea, DbH, R, p_bar,
        cat_F, rho_cat_bed, n_eff,
        Cp, stoi_mat, T0, dHrxn, F_CO0
    ) = params

    F_CO, F_H2, F_HC, F_H2O = F
    FT = np.sum(F)

    # Avoid numerical issues
    F_CO = max(F_CO, 1e-12)
    F_H2 = max(F_H2, 1e-12)
    FT = max(FT, 1e-12)

    # Mixed adiabatic temperature
    T = mixed_temperature(F, T0, Cp, dHrxn, F_CO0)

    # Temperature-dependent kinetics
    a = a0 * np.exp((Ea / R) * (1 / 493.15 - 1 / T))
    b = b0 * np.exp((DbH / R) * (1 / 493.15 - 1 / T))

    # Partial pressures
    yco = F_CO / FT
    yh2 = F_H2 / FT

    pco = yco * p_bar
    ph2 = yh2 * p_bar

    # Intrinsic and effective rates
    r_int = cat_F * a * pco * ph2 / (1 + b * pco) ** 2   # mol/(s kg_cat)
    r_eff = n_eff * r_int
    rv = r_eff * rho_cat_bed                             # mol/(s m^3)

    # Species balances
    dFdV = stoi_mat * rv

    return dFdV


# Solve only the species balances
V_span = [0, V_fin]
V_eval = np.linspace(0, V_fin, 40000)

params = [
    a0, b0, Ea, DbH, R, p_bar,
    cat_F, rho_cat_bed, n_eff,
    Cp, stoi_mat, T0, dHrxn, F_CO0
]

sol = solve_ivp(
    hybrid_pfr_mass_balance,
    V_span,
    F0,
    t_eval=V_eval,
    method="RK45",
    args=(params,)
)

# Extract results
F_CO = sol.y[0]
F_H2 = sol.y[1]
F_HC = sol.y[2]
F_H2O = sol.y[3]

# Compute temperature profile algebraically
T = np.array([mixed_temperature(sol.y[:, i], T0, Cp, dHrxn, F_CO0)
              for i in range(sol.y.shape[1])])

# Volumetric flow rate using ideal gas law
F_tot = F_CO + F_H2 + F_HC + F_H2O
Vdot = F_tot * R * T / p_pa   # m^3/s

# Concentrations
C_CO = F_CO / Vdot
C_H2 = F_H2 / Vdot
C_HC = F_HC / Vdot
C_H2O = F_H2O / Vdot

# Conversion
X = (F_CO0 - F_CO) / F_CO0


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Adiabatic Reactor', fontsize=20)

ax[0].plot(sol.t, C_CO, label="CO")
ax[0].plot(sol.t, C_H2, label="H2")
ax[0].plot(sol.t, C_H2O, label="H2O")
ax[0].plot(sol.t, C_HC, label="HC")
ax[0].set_xlabel("Reactor volume [m^3]")
ax[0].set_ylabel("Concentration [mol/m^3]")
ax[0].legend()
ax[0].grid(True)
ax[0].set_xlim(0,1)

ax[1].plot(sol.t, T)
ax[1].set_xlabel("Reactor volume [m^3]")
ax[1].set_ylabel("Temperature [K]")
ax[1].grid(True)
ax[1].set_xlim(0,1)
plt.show()

plt.figure()
plt.plot(sol.t, X)
plt.xlabel("Reactor volume [m^3]")
plt.ylabel("CO conversion [-]")
plt.grid(True)
plt.xlim(0,1)
plt.show()

# target_X = 0.8
# idx = np.argmin(np.abs(X - target_X))
# V_target = sol.t[idx]

# print(f"Volume at X = {target_X:.2f} is about {V_target:.3f} m^3")
# print(f"Temperature there is about {T[idx]:.2f} K ({T[idx]-273.15:.2f} °C)")