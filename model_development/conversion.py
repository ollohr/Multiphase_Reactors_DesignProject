import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

 

#           General constants
g = 9.81                        #gravitational acceleration                 [m/s2]
# D = 0.06                        #Diffusion coefficient                    [m2/s]
gamma = 0.375                   #Fraction of solids in the emulsion phase   [-]            idk if this should be particle hold up at mf tho... find out later (in the turbulent regime the particle hold-up is 0.3-0.45)
# K_r=1.8                         #Reaction rate constant                   [m6/(mol2*s)]
R = 8.3145                        # Gas constant [J/(K mol)]


#           Reactor Properties
V_fin = 100                             ########################## this is a temporary value, validate later if this value willl still hold 
# u =                        #Superficial gas velocity                        [m/s]                       (this would only be required for the later height and volume calculation, so fill this in after)
# mu= 1.8e-5                       #Viscosity of fluidization gas           [Pa*s]
# H_mf = 0.28                     #Bed height at u_mf                       [m]
# D_r = 0.3                       #Diameter of reactor                      [m]                          
# A_r=np.pi*D_r**2/4              #Bottom surface area of reactor           [m2]
epsilon_mf = 0.61               #Bed voidage at u_mf                        [-]


#           Gas properties
# the density of gas was recalculated for p = 20 atm and T = 340 C using --> 
rho_g = 1.94                    #Gas density                              [kg/m3]


#           Catalyst properties
rho_s = 7.794e3                   #Solid particle density                    [kg/m3]
d_p = 100e-6                     #Particle diameter                         [m]
phi_s= 1                    #Sphericity of catalyst                     [-]                     #actually idk if this is ok to assume, it was done earlier but check if this is ok


#           Parameters from the assignment
a0 = 8.88522e-3                # Reaction Rate Coefficient                  [mol /(s kg_cat bar2)]
Ea = 3.737e4                   # Activation Energy [J/mol]
b0 = 2.226                     # Adsorption coefficient at T = 493.15 K     [1/bar]
DbH = -6.837e3                 # Adsorption enthalpy                        [J/mol]

#           Input parameters
F0 = np.array([])       # this is the initial flow rate
FT = F0.sum()           # total flow rate
p = 20 * 101325                   # pressure [Pa]
T0 = 340 + 273.15      # temperature [K]
######################################################### make sure that the stoi_mat is fixed, the values here are just temporary
stoi_mat = np.array([-1,-1,1,1])                            # [H2,CO, HC, H2O] 



def reactor(V:(float),vars:float, params:np.ndarray, stoi_mat:np.ndarray)->np.ndarray:
    """
    Function ...

    ---> I think that we can model the turbulent fluidized bed reactor as a pfr, but I have to check if this is true
    Args: 
        V                    (float): volume of the reactor
        vars                 (float): vector of variables (flow rates and temperature)  
        params               (np.ndarray): vector of parameters (Ea, Ru, DbH, a0, b0, P)
        stoi_mat             (np.ndarray): stoichiometric matrix of the reaction
    
    Returns: 
        dcdV                (np.ndarray): concentration ODE over the volume of the TFB reactor
    """
    F, T = vars
    Ea, Ru, DbH, a0, b0, P = params

    FT = F.sum()
    p_CO = F[1]/FT * P 
    p_H2 = F[0]/FT * P


    alpha = a0 * np.exp((Ea/Ru)*(1/493.15 - 1/T))
    beta = b0 * np.exp((DbH/Ru)*(1/493.15 - 1/T))
    r = (F*alpha*p_CO*p_H2)/(1 + beta * p_CO)**2

    dcdV = stoi_mat.T * r
    
    return dcdV

######## Integrating the ODE
V_span = [0, V_fin]                 # decide on a final volume
V_eval = np.linspace(0,V_span[-1], 100001)

sol = solve_ivp(reactor, V_span,)

######## Conversion calculation (unless this was already done prior)



######## Plotting 