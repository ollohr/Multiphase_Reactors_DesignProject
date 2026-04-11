"""
In this model the mass balances are modeled using PFR behaviour as the gas flow is assumed to behave as plug flow.
While the temperature profile is modeled as a CSTR as the turbulence ensures that solid particles which contribute
to heat profile are perfectly mixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ---------------- General constants ----------------
g = 9.81
gamma = 0.375
R = 8.3145  # J/(K mol)

# ---------------- Reactor Properties ----------------
V_fin = 392.6990816987
u_mf = 1.19e-1
u_c = 0.87
D_r = 6.0

epsilon_mf = 0.625
epsilon_c = 0.55

# ---------------- Gas properties ----------------
rho_g = 1.94
mu = 1.54e-5

# ---------------- Catalyst properties ----------------
rho_s = 7.794e3
d_p = 100e-6
phi_s = 1.0

Wcat_per_V = 100.0
A_ht_total = 4958.5
a_ht = A_ht_total / V_fin   # m2/m3

# ---------------- Parameters from the assignment ----------------
a0 = 8.88522e-3
Ea = 3.737e4
b0 = 2.226
DbH = -6.837e3

# ---------------- Input parameters ----------------
F_CO0 = 1239.6
F_H20 = 8.0 * F_CO0
F_HC0 = 0.0
F_H2O0 = 0.0

F0 = np.array([F_H20, F_CO0, F_HC0, F_H2O0], dtype=float)
FT = F0.sum()

p = 20 * 1.01325      # bar
T0 = 340 + 273.15     # K
Tc = 250 + 273.15     # K
U = 400               # J/(m2 s K)
dHrxn = -170e3        # J/mol

Cp = np.array([29.0, 30.0, 200.0, 35.0], dtype=float)   # [H2, CO, HC, H2O]
stoi_mat = np.array([-2.0, -1.0, 1.0, 1.0], dtype=float)


def reactor(V: float, vars: np.ndarray, params: list, stoi_mat: np.ndarray) -> np.ndarray:
    """
    The mass balance is computed using a PFR model for the gas phase.
    The temperature is approximated using a CSTR-like algebraic target temperature,
    but embedded in solve_ivp through a relaxation derivative.
    """
    F = vars[:-1]
    T = vars[-1]

    # Prevent small negative trial values from RK stages
    F = np.maximum(F, 1e-12)

    Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, a_ht = params

    # ---------- Mass balance ----------
    FT = np.sum(F)
    p_CO = (F[1] / FT) * p
    p_H2 = (F[0] / FT) * p

    alpha = a0 * np.exp((Ea / R) * (1 / 493.15 - 1 / T))
    beta  = b0 * np.exp((DbH / R) * (1 / 493.15 - 1 / T))

    r_mass = (3.0 * alpha * p_CO * p_H2) / (1.0 + beta * p_CO) ** 2
    r_vol = r_mass * Wcat_per_V

    dFdV = stoi_mat * r_vol

    # ---------- Energy balance ----------
    # For a CSTR-like temperature estimate:
    # q_rxn = q_cool  ->  (-dHrxn)*r_vol = U*a_ht*(T - Tc)
    # So the algebraic target temperature is:
    T_target = Tc + ((-dHrxn) * r_vol) / (U * a_ht)

    # To keep solve_ivp structure, use a relaxation form toward the algebraic target
    dTdV = T_target - T

    return np.hstack((dFdV, dTdV))


# ---------------- Integrating the ODE ----------------
V_span = [0.0, V_fin]
V_eval = np.linspace(0.0, V_fin, 10001)

vars0 = np.append(F0, T0)
params = [Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, a_ht]

sol = solve_ivp(
    reactor,
    V_span,
    vars0,
    method="RK45",
    t_eval=V_eval,
    args=(params, stoi_mat)
)

# ---------------- Conversion calculation ----------------
X_H2 = (F0[0] - sol.y[0]) / F0[0]
X_CO = (F0[1] - sol.y[1]) / F0[1]

# ---------------- Plotting ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Molar flow rates ---
ax[0].plot(sol.t, sol.y[0], label='F_H2')
ax[0].plot(sol.t, sol.y[1], label='F_CO')
ax[0].plot(sol.t, sol.y[2], label='F_HC')
ax[0].plot(sol.t, sol.y[3], label='F_H2O')
ax[0].set_xlim(0, 20)
ax[0].legend()
ax[0].set_ylabel('Molar flow rate [mol/s]')
ax[0].set_xlabel('Reactor volume [m³]')
ax[0].set_title('Species Profiles')

# --- Temperature ---
ax[1].plot(sol.t, sol.y[4], label='Temperature', color='r')
ax[1].set_xlim(0, 20)
ax[1].legend()
ax[1].set_ylabel('Temperature [K]')
ax[1].set_xlabel('Reactor volume [m³]')
ax[1].set_title('Temperature Profile')

plt.tight_layout()
plt.show()

# ---------------- Volume at 80% conversion ----------------
X_target = 0.80
X = X_CO

if np.max(X) < X_target:
    print(f"80% conversion is not reached within V = {V_fin} m^3")
else:
    V_80 = np.interp(X_target, X, sol.t)
    
    print(f"Reactor volume at 80% conversion = {V_80:.4f} m^3")