import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


# -----------------------------
# General constants
# -----------------------------
g = 9.81
R = 8.3145  # J/(mol K)

# -----------------------------
# Reactor properties
# -----------------------------
V_fin = 392.6990816987  # m^3
u_mf = 1.19e-1
u_c = 0.87
D_r = 6.0

epsilon_mf = 0.625
epsilon_c = 0.55

# -----------------------------
# Gas properties
# -----------------------------
rho_g = 1.94
mu = 1.54e-5

# -----------------------------
# Catalyst properties
# -----------------------------
rho_s = 7.794e3
d_p = 100e-6
phi_s = 1.0

Wcat_per_V = 100.0       # kg_cat / m^3_reactor
A_ht_total = 4958.5      # m^2
a_ht = A_ht_total / V_fin  # m^2 / m^3

# -----------------------------
# Kinetic parameters
# -----------------------------
a0 = 8.88522e-3
Ea = 3.737e4
b0 = 2.226
DbH = -6.837e3

# -----------------------------
# Feed conditions
# -----------------------------
F_CO0 = 1239.6
F_H20 = 8.0 * F_CO0
F_HC0 = 0.0
F_H2O0 = 0.0

F0 = np.array([F_H20, F_CO0, F_HC0, F_H2O0], dtype=float)

p = 20 * 1.01325         # bar
T0 = 340 + 273.15        # K
Tc = 250 + 273.15        # K
U = 400.0                # W/(m^2 K) = J/(m^2 s K)
dHrxn = -170e3           # J/mol

Cp = np.array([29.0, 30.0, 200.0, 35.0], dtype=float)  # J/(mol K)
stoi_mat = np.array([-2.0, -1.0, 1.0, 1.0], dtype=float)


def kinetics(F, T, p, a0, Ea, b0, DbH, R, Wcat_per_V):
    """Return r_mass [mol/(s kg_cat)] and r_vol [mol/(s m^3_reactor)]."""
    FT = np.sum(F)
    if FT <= 0:
        return 0.0, 0.0

    y_H2 = max(F[0], 0.0) / FT
    y_CO = max(F[1], 0.0) / FT

    p_H2 = y_H2 * p  # bar
    p_CO = y_CO * p  # bar

    alpha = a0 * np.exp((Ea / R) * (1.0 / 493.15 - 1.0 / T))
    beta = b0 * np.exp((DbH / R) * (1.0 / 493.15 - 1.0 / T))

    # activity factor = 3
    r_mass = 3.0 * alpha * p_CO * p_H2 / (1.0 + beta * p_CO) ** 2
    r_vol = r_mass * Wcat_per_V
    return r_mass, r_vol


def cstr_temperature(F, V, T_in, Tc, U, A_ht_total, dHrxn,
                     p, a0, Ea, b0, DbH, R, Wcat_per_V, Cp):
    """
    Solve the algebraic steady-state CSTR energy balance:
    0 = sum(F_i Cp_i)(T_in - T) + (-dHrxn) r_vol(T) V + UA(Tc - T)

    Returns a physically reasonable temperature even if a simple
    bisection bracket is not available.
    """
    F = np.maximum(F, 1e-12)   # prevent negative/zero trial values
    Cp_flow = np.dot(F, Cp)    # J/(s K)

    def residual(T):
        _, r_vol = kinetics(F, T, p, a0, Ea, b0, DbH, R, Wcat_per_V)
        return Cp_flow * (T_in - T) + (-dHrxn) * r_vol * V + U * A_ht_total * (Tc - T)

    # Try to find a bracket automatically
    T_grid = np.linspace(300.0, 1600.0, 200)
    f_grid = np.array([residual(T) for T in T_grid])

    # Search for a sign change
    for i in range(len(T_grid) - 1):
        if np.isnan(f_grid[i]) or np.isnan(f_grid[i + 1]):
            continue
        if f_grid[i] == 0:
            return T_grid[i]
        if f_grid[i] * f_grid[i + 1] < 0:
            sol = root_scalar(
                residual,
                bracket=[T_grid[i], T_grid[i + 1]],
                method="bisect"
            )
            if sol.converged:
                return sol.root

    # Fallback: choose temperature with smallest absolute residual
    idx = np.argmin(np.abs(f_grid))
    return T_grid[idx]


def reactor(V, F, params, stoi_mat):
    """
    Species-only ODE system.
    Temperature is computed algebraically from a steady-state CSTR energy balance.
    """
    Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, A_ht_total, T0, Cp = params

    # Algebraic temperature from CSTR energy balance
    T = cstr_temperature(
        F=F, V=V, T_in=T0, Tc=Tc, U=U, A_ht_total=A_ht_total, dHrxn=dHrxn,
        p=p, a0=a0, Ea=Ea, b0=b0, DbH=DbH, R=R, Wcat_per_V=Wcat_per_V, Cp=Cp
    )

    # Reaction rate at that temperature
    _, r_vol = kinetics(F, T, p, a0, Ea, b0, DbH, R, Wcat_per_V)

    # PFR species balances
    dFdV = stoi_mat * r_vol
    return dFdV


# -----------------------------
# Integrate species balances
# -----------------------------
V_span = [0.0, V_fin]
V_eval = np.linspace(0.0, V_fin, 2000)

params = [Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, A_ht_total, T0, Cp]

sol = solve_ivp(
    reactor,
    V_span,
    F0,
    method="RK45",
    t_eval=V_eval,
    args=(params, stoi_mat),
    rtol=1e-6,
    atol=1e-9
)

if not sol.success:
    raise RuntimeError(sol.message)

# -----------------------------
# Post-process temperature
# -----------------------------
T_profile = np.array([
    cstr_temperature(
        F=sol.y[:, i], V=sol.t[i], T_in=T0, Tc=Tc, U=U, A_ht_total=A_ht_total, dHrxn=dHrxn,
        p=p, a0=a0, Ea=Ea, b0=b0, DbH=DbH, R=R, Wcat_per_V=Wcat_per_V, Cp=Cp
    )
    for i in range(sol.t.size)
])

# -----------------------------
# Conversion
# -----------------------------
X_H2 = (F0[0] - sol.y[0]) / F0[0]
X_CO = (F0[1] - sol.y[1]) / F0[1]

# -----------------------------
# Plotting
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(sol.t, sol.y[0], label='F_H2')
ax[0].plot(sol.t, sol.y[1], label='F_CO')
ax[0].plot(sol.t, sol.y[2], label='F_HC')
ax[0].plot(sol.t, sol.y[3], label='F_H2O')
ax[0].set_xlim(0, V_fin)
ax[0].legend()
ax[0].set_ylabel('Molar flow rate [mol/s]')
ax[0].set_xlabel('Reactor volume [m³]')
ax[0].set_title('Species Profiles')

ax[1].plot(sol.t, T_profile, label='Temperature', color='r')
ax[1].set_xlim(0, V_fin)
ax[1].legend()
ax[1].set_ylabel('Temperature [K]')
ax[1].set_xlabel('Reactor volume [m³]')
ax[1].set_title('Temperature Profile')

plt.tight_layout()
plt.show()

# -----------------------------
# Volume at 80% conversion
# -----------------------------
X_target = 0.80
X = X_CO

if np.max(X) < X_target:
    print(f"80% conversion is not reached within V = {V_fin:.3f} m^3")
else:
    V_80 = np.interp(X_target, X, sol.t)
    print(f"Reactor volume at 80% conversion = {V_80:.4f} m^3")