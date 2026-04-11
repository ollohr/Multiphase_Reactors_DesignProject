import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
R = 8.3145  # J/(mol K)


# Reactor conditions
T0 = 340 + 273.15           # K
p_bar = 20 * 1.01325        # bar
p_pa = p_bar * 1e5          # Pa

# Bed properties
epsilon_c = 0.55
rho_s = 7.794e3             # kg/m^3
rho_cat_bed = (1 - epsilon_c) * rho_s

cat_F = 1.0
n_eff = 0.9

# Kinetic parameters
a0 = 8.88522e-3             # mol/(s kg_cat bar^2)
Ea = 3.737e4                # J/mol
b0 = 2.226                  # 1/bar
DbH = -6.837e3              # J/mol

# Reaction enthalpy (per mol CO reacted)
dHrxn = -170e3              # J/mol CO reacted

# Approximate constant heat capacities [J/(mol K)]
# Use better values if your course/material provides them
Cp_CO = 30.0
Cp_H2 = 29.0
Cp_HC = 60.0
Cp_H2O = 40.0

# Inlet molar flows [mol/s]
F_CO0 = 1239.6
F_H20 = 8.0 * F_CO0
F_HC0 = 0.0
F_H2O0 = 0.0

def dY_dV(V, Y):
    F_CO, F_H2, F_HC, F_H2O, T = Y

    # Prevent nonphysical values
    F_CO = max(F_CO, 1e-12)
    F_H2 = max(F_H2, 1e-12)
    F_HC = max(F_HC, 0.0)
    F_H2O = max(F_H2O, 0.0)
    T = max(T, 250.0)

    # Temperature-dependent kinetics
    a = a0 * np.exp(Ea / R * (1/493.15 - 1/T))
    b = b0 * np.exp(DbH / R * (1/493.15 - 1/T))

    F_tot = F_CO + F_H2 + F_HC + F_H2O

    yco = F_CO / F_tot
    yh2 = F_H2 / F_tot

    pco = yco * p_bar
    ph2 = yh2 * p_bar

    # Intrinsic and effective rates
    r_int = cat_F * a * pco * ph2 / (1 + b * pco)**2   # mol/(s kg_cat)
    r = n_eff * r_int
    rv = r * rho_cat_bed                               # mol/(s m^3)

    # Species balances
    dFCO_dV = -rv
    dFH2_dV = -2.0 * rv
    dFHC_dV = rv
    dFH2O_dV = rv

    # Adiabatic energy balance
    Cp_flow = (
        F_CO * Cp_CO +
        F_H2 * Cp_H2 +
        F_HC * Cp_HC +
        F_H2O * Cp_H2O
    )  # J/(s K)

    dT_dV = ((-dHrxn) * rv) / Cp_flow   # K/m^3

    return [dFCO_dV, dFH2_dV, dFHC_dV, dFH2O_dV, dT_dV]

V_span = [0, 1]
V_eval = np.linspace(0, 1, 400)

sol = solve_ivp(
    dY_dV,
    V_span,
    [F_CO0, F_H20, F_HC0, F_H2O0, T0],
    t_eval=V_eval,
    method="BDF"   # often more stable for strong exotherms
)

F_CO = sol.y[0]
F_H2 = sol.y[1]
F_HC = sol.y[2]
F_H2O = sol.y[3]
T = sol.y[4]

F_tot = F_CO + F_H2 + F_HC + F_H2O
Vdot = F_tot * R * T / p_pa   # m^3/s

C_CO = F_CO / Vdot
C_H2 = F_H2 / Vdot
C_HC = F_HC / Vdot
C_H2O = F_H2O / Vdot

X = (F_CO0 - F_CO) / F_CO0

plt.figure()
plt.plot(sol.t, C_CO, label="CO")
plt.plot(sol.t, C_H2, label="H2")
plt.plot(sol.t, C_H2O, label="H2O")
plt.plot(sol.t, C_HC, label="HC")
plt.xlabel("Reactor volume [m^3]")
plt.ylabel("Concentration [mol/m^3]")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(sol.t, T)
plt.xlabel("Reactor volume [m^3]")
plt.ylabel("Temperature [K]")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(sol.t, X)
plt.xlabel("Reactor volume [m^3]")
plt.ylabel("CO conversion [-]")
plt.grid(True)
plt.show()

target_X = 0.8
idx = np.argmin(np.abs(X - target_X))
V_target = sol.t[idx]

print(f"Volume at X = {target_X:.2f} is about {V_target:.3f} m^3")
print(f"Temperature there is about {T[idx]:.2f} K ({T[idx]-273.15:.2f} °C)")