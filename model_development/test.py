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
D_r = 6.0

epsilon_mf = 0.625               #Bed voidage at u_mf                        [-]
epsilon_c  = 0.55                #Bed voidage at u_c                         [-]


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
# rho_cat_bed = (1-epsilon_c)*rho_s
Wcat_per_V = 100.0          # kg_cat / m3_reactor, temporary design value
A_ht_total = 4958.5         # m2, from your report
a_ht = A_ht_total / V_fin   # m2/m3, temporary since V_fin is still provisional

#           Parameters from the assignment
a0 = 8.88522e-3                # Reaction Rate Coefficient                  [mol /(s kg_cat bar2)]
Ea = 3.737e4                   # Activation Energy [J/mol]
b0 = 2.226                     # Adsorption coefficient at T = 493.15 K     [1/bar]
DbH = -6.837e3                 # Adsorption enthalpy                        [J/mol]

#           Input parameters
F_CO0 = 1239.6                  # [mol/s]
F_H20 = 8.0 * F_CO0             # [mol/s]
F_HC0 = 0.0                     # [mol/s]
F_H2O0 = 0.0                    # [mol/s]

F0 = np.array([F_H20, F_CO0, F_HC0, F_H2O0])        
FT = F0.sum()           # total molar flow rate
p = 20*1.01325                   # pressure for kinetics (units) [bar]
T0 = 340 + 273.15             # temperature [K]
Tc = 250 + 273.15             # cooling water temperature [K]
U = 400                       # heat transfer coefficient [J/(m2 s K)]
dHrxn = -170e3                # enthalpy of reaction [J/mol]

Cp = np.array([29.0, 30.0, 200.0, 35.0])   # [H2, CO, HC, H2O] in [J/(mol K)]           # (make sure they are correct first) write these in the report later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
stoi_mat = np.array([-2.0,-1.0,1.0,1.0])                            # [H2,CO, HC, H2O] 



def reactor(V:(float),vars:float, params:np.ndarray, stoi_mat:np.ndarray)->np.ndarray:
    """
    Function ...

    ---> I think that we can model the turbulent fluidized bed reactor as a pfr, but I have to check if this is true
    Args: 
        V                    (float): volume of the reactor
        vars                 (float): vector of variables (flow rates and temperature)  
        params               (np.ndarray): vector of parameters (Ea, R, DbH, a0, b0, P)
        stoi_mat             (np.ndarray): stoichiometric matrix of the reaction
    
    Returns: 
        dcdV                (np.ndarray): concentration ODE over the volume of the TFB reactor
    """
    F = vars[:-1]
    T = vars[-1]

    Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, a_ht = params

    # mass balance 
    FT = F.sum()
    p_CO = F[1]/FT * p 
    p_H2 = F[0]/FT * p

    alpha = a0 * np.exp((Ea/R)*(1/493.15 - 1/T))
    beta = b0 * np.exp((DbH/R)*(1/493.15 - 1/T))

    r_mass = (3*alpha*p_CO*p_H2)/(1 + beta * p_CO)**2        # taking activity of catalyst to be 3
    r_vol = r_mass* Wcat_per_V

    dFdV = stoi_mat * r_vol

    # energy balance 
    Cp_flow = F@Cp
    # dTdV = (dHrxn * dFdV + U * (Tc - T))/(Cp_flow)
    q_rxn = (-dHrxn) * r_vol
    q_cool = U * a_ht* (Tc - T)
    dTdV = (q_rxn + q_cool) / Cp_flow                   
    
    return np.hstack((dFdV, dTdV))

######## Integrating the ODE
V_span = [0, V_fin]                 # decide on a final volume
V_eval = np.linspace(0,V_span[-1], 100001)

vars = np.append(F0, T0)
params  = [Ea, R, DbH, dHrxn, U, Tc, a0, b0, p, Wcat_per_V, F0, a_ht]
sol = solve_ivp(reactor, V_span, vars, method = "RK45", t_eval = V_eval, args= (params, stoi_mat))


######## Conversion calculation (unless this was already done prior)
X_H2 = (F0[0] - sol.y[0])/F0[0]
X_CO = (F0[1] - sol.y[1])/F0[1]



######## Plotting 
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Molar flow rates ---
ax[0].plot(sol.t, sol.y[0], label='F_H2')
ax[0].plot(sol.t, sol.y[1], label='F_CO')
ax[0].plot(sol.t, sol.y[2], label='F_HC')
ax[0].plot(sol.t, sol.y[3], label='F_H2O')

ax[0].legend()
ax[0].set_ylabel('Molar flow rate [mol/s]')
ax[0].set_xlabel('Reactor volume [m³]')
ax[0].set_title('Species Profiles')

# --- Temperature ---
ax[1].plot(sol.t, sol.y[4], label='Temperature', color='r')

ax[1].legend()
ax[1].set_ylabel('Temperature [K]')
ax[1].set_xlabel('Reactor volume [m³]')
ax[1].set_title('Temperature Profile')

plt.tight_layout()
plt.show()




