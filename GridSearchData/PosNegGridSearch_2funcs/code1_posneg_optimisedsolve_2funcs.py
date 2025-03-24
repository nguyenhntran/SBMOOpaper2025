#  ---------------------------------------------------------------------
# |                                                                     |
# |                   PART 1: CREATING VALUES TO PLOT                   |
# |                                                                     |
#  ---------------------------------------------------------------------

# _________ Python preliminary _________
                                       #|
import numpy as np                     #|
from paretoset import paretoset        #|
from scipy.optimize import fsolve      #|
import sys                             #|

# Define leakiness value
L = 1


# ____________________________ DEFINE DYNAMICAL SYSTEM _______________________________
#                               (CAN EDIT THIS CELL)                                 #|                             
                                                                                     #|
# x-nullcline                                                                        #|
def Equ1(x, y, beta_x, n):                                                           #|
    return beta_x / (1 + y**n) - x + L
                                                                                     #|
# y-nullcline                                                                        #|
def Equ2(x, y, beta_y, n):                                                           #|
    return beta_y * x**n / (1 + x**n) - y + L
                                                                                     #|
# ____________________________________________________________________________________

print('checkpoint1')

# ________________ FUNCTION THAT EVALUATES VECTOR FIELD AT A POINT ___________________
                                                                                     #|
# Function that takes in coordinate   P = [x value, y value],                        #|
#                        initial time t = time value,                                #|
#                        parameters   params = [param 1 value, param 2 value, ...]   #|
# and returns corresponding value of dx/dt and dy/dt in array of form [dx/dt, dy/dt] #|
def Equs(P, t, params):                                                              #|
                                                                                     #|
    x = P[0]                                                                         #|
    y = P[1]                                                                         #|
                                                                                     #|
    beta_x = params[0]                                                               #|
    beta_y = params[1]                                                               #|
    n      = params[2]                                                               #|
                                                                                     #|
    val0 = Equ1(x, y, beta_x, n)                                                     #|
    val1 = Equ2(x, y, beta_y, n)                                                     #|
                                                                                     #|
    return np.array([val0, val1])                                                    #|
                                                                                     #|
# ____________________________________________________________________________________

print('checkpoint2')

# __________________________ PAMATER RANGES OF INTEREST ____________________________
#                               CAN EDIT THIS CELL                                 #|
                                                                                   #|
beta_x_min  = 0.01                                                                 #|
beta_x_max  = 50                                                                   #|
beta_x_no   = 1000                                                                 #|
beta_x_vals = np.linspace(beta_x_min,beta_x_max,beta_x_no)                         #|
                                                                                   #|
#We are only taking 1 betax at a time                                              #|
betaindex = int(sys.argv[1])                                                       #|
beta_x_vals = np.array([beta_x_vals[betaindex]])                                   #|
beta_x_no = 1                                                                      #|
                                                                                   #|
beta_y_min  = 0.01                                                                 #|
beta_y_max  = 50                                                                   #|
beta_y_no   = 1000                                                                 #|
beta_y_vals = np.linspace(beta_y_min,beta_y_max,beta_y_no)                         #|
                                                                                   #|
n_min  = 0.01                                                                      #|
n_max  = 10                                                                        #|
n_no   = 1000                                                                      #|
n_vals = np.linspace(n_min,n_max,n_no)                                             #|
                                                                                   #|
# __________________________________________________________________________________

print('checkpoint3')

# ________________ TABULATE OUTER CUBE WHILE FILTERING OUT REGIONS OF CUBE WE DO NOT WANT _________________
#                        IE. CREATING PARAMTER POLYHEDRON SUBSPACE OF INTEREST                            #|
                                                                                                          #|
#  ---------------------------------------                                                                #|
# | beta_x value | beta_y value | n value |                                                               #|
# |       #      |       #      |    #    | <--- row 0                                                    #|
# |       #      |       #      |    #    |                                                               #|
# |       #      |       #      |    #    |                                                               #|
# |       #      |       #      |    #    | <--- row (beta_x_no)*(beta_y_no)*(n_no)                       #|
#  ---------------------------------------                                                                #|
                                                                                                          #|
# Initialise memory corresponding to largest case scenario                                                #|
ParamCombinations = np.full( (beta_x_no*beta_y_no*n_no , 3) , None)                                       #|
                                                                                                          #|
# Dummy counter to track current row in table                                                             #|
currentrow = 0                                                                                            #|
                                                                                                          #|
# For each position in parameter space cube                                                               #|
for beta_x_val in beta_x_vals:                                                                            #|
  for beta_y_val in beta_y_vals:                                                                          #|
    for n_val in n_vals:                                                                                  #|
        ParamCombinations[currentrow,:] = np.array([beta_x_val, beta_y_val, n_val])                       #|
        # Update current row                                                                              #|
        currentrow += 1                                                                                   #|
                                                                                                          #|
# _________________________________________________________________________________________________________

print('checkpoint4')

# __________________________________________ TABLE OF X AND Y STEADY STATES _________________________________________
#                                        CAN EDIT INITGUESSES ARRAY IN THIS CELL                                    #|
#                                                                                                                   #|
#             ---------                               ---------                                                     #|
#            |   xss   |                             |   yss   |                                                    #|
#            |    #    |                             |    #    |                                                    #|
# xss1   =   |    #    |        and       yss1   =   |    #    |                                                    #|
#            |    #    |                             |    #    |                                                    #|
#            |    #    |                             |    #    |                                                    #|
#             ---------                               ---------                                                     #|
                                                                                                                    #|
# Get number of rows in table                                                                                       #|
rows = ParamCombinations.shape[0]                                                                                   #|
                                                                                                                    #|
# Initialize empty arrays to store steady-state values (xss and yss) 
xss1 = np.empty((rows, 1))  # To store the first x steady state for each row                                        #|
yss1 = np.empty((rows, 1))  # To store the second x steady state for each row                                       #|
                                                                                                                    #|
# For each position in parameter polyhedron                                                                         #|
for row in range(rows):                                                                                             #|
                                                                                                                    #|
    # Extract the parameter values (beta_x_val, beta_y_val, and n_val) from the current row                         #|
    beta_x_val = ParamCombinations[row, 0]                                                                          #|
    beta_y_val = ParamCombinations[row, 1]                                                                          #|
    n_val      = ParamCombinations[row, 2]                                                                          #|
                                                                                                                    #|
    # Store the parameter values in an array for passing to the equation solver                                     #|
    params = np.array([beta_x_val, beta_y_val, n_val])                                                              #|
                                                                                                                    #|
    # Initial guesses for solving the steady-state equations                                                        #|
    InitGuesses = [
    np.array([beta_x_val, beta_y_val]),
    np.array([beta_x_val/2, beta_y_val/2]),
    np.array([beta_x_val, 0]),
    np.array([beta_x_val/2, 0]),
    np.array([0, beta_y_val]),
    np.array([0, beta_y_val/2]),
    np.array([1,1]),
    np.array([0.1,0.1]),
    np.array([0,0]),
    np.array([0.3,12.5]),
    np.array([50.5,0.9]),
    np.array([67.6,0.9]),
    np.array([21,0.9])]
                                                                                                                    #|
    # To store valid solutions                                                                                      #|
    solutions = []                                                                                                  #|
                                                                                                                    #|
    # Iterate over the initial guesses and solve the equations                                                      #|
    for InitGuess in InitGuesses:                                                                                   #|
        # Solve the steady-state equations using fsolve                                                             #|
        t = 0.0                                                                                                     #|
        output, infodict, intflag, mesg = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)   #|
        # Extract x and y steady states                                                                             #|
        xss, yss = output                                                                                           #|
        # Residuals from fsolve (how well the solution satisfies the equations)                                     #|
        fvec = infodict['fvec']                                                                                     #|
                                                                                                                    #|
        # Check stability of steady state                                                                           #|
        delta = 1e-8                                                                                                #|
        dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta                                #|
        dEqudy = (Equs([xss,yss+delta], t, params)-Equs([xss,yss], t, params))/delta                                #|
        jac = np.transpose(np.vstack((dEqudx,dEqudy)))                                                              #|
        eig = np.linalg.eig(jac)[0]                                                                                 #|
        instablility = np.any(np.real(eig) >= 0)                                                                    #|
                                                                                                                    #|
        # Check conditions for valid steady states                                                                  #|
        # i.e. both xss and yss large enough, residuals small, and successful convergence                           #|
        if xss > 0.04 and yss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:     #|
            xss1[row, 0] = xss
            yss1[row, 0] = yss
        else:
            xss1[row, 0] = float('nan')
            yss1[row, 0] = float('nan') 
                                                                                                                    #|
#____________________________________________________________________________________________________________________

print('checkpoint5')

np.savez('posneg_steadystates' + str(betaindex) + '.npz', xss1=xss1, yss1=yss1)

print('checkpoint6')

# ___________________________________ DEFINE SENSITIVITY FUNCTIONS ___________________________________
#                                          CAN EDIT THIS CELL                                        #|
                                                                                                     #|
# Define analytical expression for s_betax(xss)                                                      #|
def S_betax_xss_analytic(xss, yss, beta_x, beta_y, n):                                               #|
    numer = (- L + xss) * beta_x * beta_y
    denom = xss * beta_x * beta_y + n**2 * (L - xss)**2 * xss**(-1 + n) * yss**(-1 + n) * (L - yss + beta_y)**2 * beta_x * beta_y
    sensitivity = numer/denom    
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
# Define analytical expression for s_betay(xss)                                                      #|
def S_betay_xss_analytic(xss, yss, beta_x, beta_y, n):                                               #|
    numer = n * (L - xss) * (L - yss) * (L - xss + beta_x) * (beta_y**2)
    denom = (L - yss + beta_y) * (n**2 * (L - xss)**2 * (L - yss) * beta_y + beta_x * (L * xss * (1 + xss**n)**2 * (L - yss) + (L**2 * n**2 - xss * (-n**2 + xss**n + xss**(2 * n)) * yss + L * (xss - n**2 * xss + 3 * xss**(1 + n) + 2 * xss**(1 + 2 * n) - n**2 * yss)) * beta_y + xss**(1 + n) * (1 + xss**n) * (beta_y**2)))
    sensitivity = numer/denom                                                                      #|
    return abs(sensitivity)                                                                          #|
                                                                                                     #|
# ____________________________                                                                       #|
                                                                                                     #|
                                                                                                     #|
#_____________________________________________________________________________________________________

print('checkpoint7')

# _______________________ TABLE OF SENSITIVITIES FOR STEADY STATES 1 AND 2 ____________________________
#                                                                                                     #|
#                                                                                                     #|
# Create two empty sensitivity space numpy array of shape                                             #|
#  -------------------------------------------------------------------------------------------------  #|
# | S_{beta_x}(xss) | S_{beta_x}(yss) |
# |         #       |         #       |
# |         #       |         #       |
# |         #       |         #       |   
# |         #       |         #       | 
#  -------------------------------------------------------------------------------------------------  #|
Sens1 = np.empty([rows, 2])                                                                           #|
                                                                                                      #|
# For each row in parameter table                                                                     #|
for row in range(rows):                                                                               #|
                                                                                                      #|
      # get the corresponding parameter values                                                        #|
      beta_x_val = ParamCombinations[row,0]                                                           #|
      beta_y_val = ParamCombinations[row,1]                                                           #|
      n_val      = ParamCombinations[row,2]                                                           #|
                                                                                                      #|
      # get the corresponding steady state 1 values                                                   #|
      xss1_val = xss1[row]                                                                            #|
      yss1_val = yss1[row]                                                                            #|
                                                                                                      #|
      # compute sensitivity values of steady state 1                                                  #|
      S_beta_x_xss_val1 = S_betax_xss_analytic(xss1_val, yss1_val, n_val, beta_x_val, beta_y_val)     #|
      S_beta_y_xss_val1 = S_betay_xss_analytic(xss1_val, yss1_val, n_val, beta_x_val, beta_y_val)     #|
                                                                                                      #|
      # Add sensitivity values to sensitivity table 1                                                 #|
      Sens1[row,:] = np.array([S_beta_x_xss_val1,                                                     #|
                               S_beta_y_xss_val1]).flatten()                                          #|
                                                                                                      #|
#______________________________________________________________________________________________________

print('checkpoint8')

# _______________________ PARETO FRONTS FOR EACH PAIR OF SENSITIVITY POLYHEDRONS _________________________
                                                                                                         #|
# There may be NaNs in the array. Pareto minimisation will think NaNs                                    #|
# are minimum. We don't want this. Let's replace NaNs with infinities.                                   #|
Sens1 = np.where(np.isnan(Sens1), np.inf, Sens1)                                                         #|
                                                                                                         #|
# _____________________________________________________________________                                  #|
                                                                                                         #|
# Create a new table for each unique pair of sensitivites                                                #|
Sens1Pair1  = Sens1[:, [0, 1]]      #Columns: 'S_beta_x_xss', 'S_beta_x_yss'                             #|
                                                                                                         #|
# _____________________________________________________________________                                  #|
                                                                                                         #|
# Run Pareto tool with minimisation setting to get a mask.                                               #|
# Each mask is an array of the form [True, False, True, ...].                                            #|
# Indexing another array with this mask will remove rows corresponding to Falses.                        #|
# Eg:  dummy       = [1   , 2   , 3    ]                                                                 #|
#      mask        = [True, True, False]                                                                 #|
#      dummy[mask] = [1   , 2          ]                                                                 #|
                                                                                                         #|
Sens1mask1 = paretoset(Sens1Pair1, sense=["min", "min"])                                                 #|
paretoset_Sens1Pair1 = Sens1Pair1[Sens1mask1]                                                            #|
                                                                                                         #|
                                                                                                         #|
# Save tables of Pareto points for each sensitivity pair                                                 #|
np.savez('posneg_Paretos_Sens1' + str(betaindex) + '.npz',                                               #|
                                                        paretoset_Sens1Pair1  = paretoset_Sens1Pair1)    #|
                                                                                                         #|
                                                                                                         #|
#_________________________________________________________________________________________________________

print('checkpoint9')

# ___________________ CORRESPONDING PARETO FRONTS IN PARAMETER SPACE _____________________
                                                                                         #|
# Get the corresponding pareto fronts in parameter space                                 #|
                                                                                         #|
paretoset_Sens1mask1_Param  = ParamCombinations[Sens1mask1]                              #|
                                                                                         #|
# ______________                                                                         #|
                                                                                         #|
# Save the arrays                                                                        #|
                                                                                         #|
np.savez('posneg_Paretos_Params1' + str(betaindex) + '.npz',                             #|
			paretoset_Sens1mask1_Param  = paretoset_Sens1mask1_Param)                    #|
                                                                                         #|
#_________________________________________________________________________________________

' + str(betaindex) + '
' + str(betaindex) + '

print('checkpoint10')