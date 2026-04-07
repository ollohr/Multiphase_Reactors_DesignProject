import numpy as np
import matplotlib.pyplot as plt


######## parameters 
# parameters which have been commented out still need to be found (these are all unknonw)

g = 9.81                        #gravitational acceleration                 [m/s2]
rho_s = 7.794                    #Solid particle density                    [kg/m3]
rho_g = 0.2064                    #Gas density                              [kg/m3]
u =                        #Superficial gas velocity                        [m/s]                       (this would only be required for the later height and volume calculation, so fill this in after)
# mu= 1.8e-5                       #Viscosity of fluidization gas           [Pa*s]
# H_mf = 0.28                     #Bed height at u_mf                       [m]
d_p = 100e-4                     #Particle diameter                         [m]
# D_r = 0.3                       #Diameter of reactor                      [m]                          
# A_r=np.pi*D_r**2/4              #Bottom surface area of reactor           [m2]
# D = 0.06                        #Diffusion coefficient                    [m2/s]
gamma = 0.375                   #Fraction of solids in the emulsion phase   [-]            idk if this should be particle hold up at mf tho... find out later (in the turbulent regime the particle hold-up is 0.3-0.45)
# K_r=1.8                         #Reaction rate constant                   [m6/(mol2*s)]
epsilon_mf = 0.61               #Bed voidage at u_mf                        [-]
phi_s= 0.83                     #Sphericity of catalyst                     [-]
C_b0= 5                         #Concentration in feed                      [mol/m3]


### the parameters from the assignment
a0 = 8.88522e-3                # Reaction Rate Coefficient                  [mol /(s kg_cat bar2)]
Ea = 3.737e4                   # Activation Energy [J/mol]
b0 = 2.226                     # Adsorption coefficient at T = 493.15 K     [1/bar]
DbH = -6837e3                  # Adsorption enthalpy                        [J/mol]

######## model 

stoi_mat = np.array([-])

def reactor(t,x):
    """
    Function ...

    Args: 
        ...
    
    Returns: 
        ...
    """

    return



######## Conversion calculation (unless this was already done prior)



######## Plotting 