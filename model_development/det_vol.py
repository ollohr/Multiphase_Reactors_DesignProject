import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

 

#           General constants
g = 9.81                        #gravitational acceleration                 [m/s2]
gamma = 0.375                   #Fraction of solids in the emulsion phase   [-]            idk if this should be particle hold up at mf tho... find out later (in the turbulent regime the particle hold-up is 0.3-0.45)
R = 8.3145                        # Gas constant [J/(K mol)]

# K_r=1.8                         #Reaction rate constant                   [m6/(mol2*s)]
# D = 0.06                        #Diffusion coefficient                    [m2/s]


#           Reactor Properties
V_fin = 100                             # temporary value for integration might need to be changed later [m3]
u_mf = 1.19e-1                          # minimum fluidization velocity [m/s]
u_c = 0.87                              # turbulent fluidization velocity [m/s]

epsilon_mf = 0.625               #Bed voidage at u_mf                        [-]
epsilon_c  = 0.55                #Bed voidage at u_c                         [-]
T = 340 + 273.15                         # reactor temp
p = 20*1.01325                   # pressure for kinetics (units) [bar]
pa= p * 1e5                   # pressure [Pa]



# u =                        #Superficial gas velocity                        [m/s]                       (this would only be required for the later height and volume calculation, so fill this in after)
# A_r=np.pi*D_r**2/4              #Bottom surface area of reactor           [m2]
# H_mf = 0.28                     #Bed height at u_mf                       [m]
# D_r = 0.3                       #Diameter of reactor                      [m]                          

#           Gas properties
rho_g = 1.94                            #Gas density                              [kg/m3]
mu= 1.54e-5                             # Dynamic viscosity of gas           [Pa*s]


#           Catalyst properties
rho_s = 7.794e3                   #Solid particle density                    [kg/m3]
d_p = 100e-6                     #Particle diameter                         [m]
phi_s= 1                    #Sphericity of catalyst                     [-]                     #actually idk if this is ok to assume, it was done earlier but check if this is ok
rho_cat_bed = (1-epsilon_c)*rho_s
cat_F = 1                   # catalyst factor
n_eff = 0.9

#           Parameters from the assignment
a0 = 8.88522e-3                # Reaction Rate Coefficient                  [mol /(s kg_cat bar2)]
Ea = 3.737e4                   # Activation Energy [J/mol]
b0 = 2.226                     # Adsorption coefficient at T = 493.15 K     [1/bar]
DbH = -6.837e3                 # Adsorption enthalpy                        [J/mol]
a = a0*np.exp(Ea/R*(1/493.15-1/T))
b = b0*np.exp(DbH/R*(1/493.15-1/T))


#           Input parameters
F_CO0 = 1239.6                  # [mol/s]
F_H20 = 8.0 * F_CO0             # [mol/s]
F_HC0 = 0.0                     # [mol/s]
F_H2O0 = 0.0                    # [mol/s]

#### determining Reaction Volume #####

def dX_dV(V, X):
    X = X[0]   

    F_CO = F_CO0 * (1 - X)
    F_H2 = F_H20 - 2 * F_CO0 * X
    F_HC = F_CO0 * X
    F_H2O = F_CO0 * X

    F_tot = F_CO + F_H2 + F_HC + F_H2O

    yco = F_CO / F_tot
    yh2 = F_H2 / F_tot

    pco = yco * p
    ph2 = yh2 * p

    r_int = cat_F * a * pco * ph2 / (1 + b * pco)**2
    r = n_eff * r_int
    rv = r * rho_cat_bed

    return [rv / F_CO0]

V_span = [0, 100]               
V_eval = np.linspace(0, 100, 200)

sol = solve_ivp(dX_dV, V_span, [0.0], t_eval=V_eval)

plt.plot(sol.t, sol.y[0])
plt.xlabel("Reactor volume V [m^3]")
plt.ylabel("CO conversion X [-]")
plt.grid(True)
# plt.show()

target_X = 0.8
idx = np.argmin(np.abs(sol.y[0] - target_X))
V_target = sol.t[idx]

print(f"Volume at X = {target_X:.2f} is about {V_target:.2f} m^3")

#