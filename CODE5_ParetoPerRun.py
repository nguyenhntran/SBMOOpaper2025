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
#______________________________________#|


# -------------- PART 0a: CHOOSE CIRCUIT AND SET UP FOLDER --------------                            #<--------- SAME AS CODE1 UP TO...


# Choose circuit
circuit = input("Please enter name of the circuit: ")

# Import circuit config file
config = importlib.import_module(circuit)

# Define subfolder name to work in
folder_name = f"MOSA_{circuit}"

# Create folder if not yet exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Jump to folder
os.chdir(folder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0b: DEFINE DYNAMICAL SYSTEM --------------


# dx/dt
Equ1 = config.Equ1
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    alpha = params[0]
    n     = params[1]
    val0 = Equ1(x, alpha, n)
    return np.array([val0])

# Define initial time
t = 0.0

# Define number of steady states expected
numss = int(input("""
Do you expect 1 or 2 stable steady states in your search space? 
Please enter either 1 or 2: """))


# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------


# Load analytical sensitivity expressions
S_alpha_xss_analytic = config.S_alpha_xss_analytic
S_n_xss_analytic = config.S_n_xss_analytic


# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------


# Print prompt
print("""
Only two sensitivity functions are present:
0. |S_alpha_xss|
1. |S_n_xss|
MOSA will anneal this pair.
""")

# Choose pair of functions
choice1 = 0
choice2 = 1

# List of sensitivity function names
sensitivity_labels = [
    "|S_alpha_xss|",
    "|S_n_xss|"]

# Save function names for later use
label1 = sensitivity_labels[choice1]
label2 = sensitivity_labels[choice2]


# -------------- PART 0e: CHANGING DIRECTORIES --------------


# Define subfolder name to work in
subfolder_name = f"MOSA_sensfuncs_{choice1}_and_{choice2}"

# Create folder if not yet exist
if not os.path.exists(subfolder_name):
    os.makedirs(subfolder_name)

# Jump to folder
os.chdir(subfolder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0f: DEFINE FUNCTIONS --------------


# DEFINE FUNCTION TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def euclidean_distance(x1, x2):
    return abs(x1 - x2)

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS GIVEN AN INITIAL GUESS
def ssfinder(alpha_val,n_val):

    # If we have one steady state
    if numss == 1:

        # Load initial guesses for solving which can be a function of a choice of alpha and n values
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)

        # Define array of parameters
        params = np.array([alpha_val, n_val])

        # For each initial guess in the list of initial guesses we loaded
        for InitGuess in InitGuesses:
            
            # Get solution details
            output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)
            xss = output
            fvec = infodict['fvec']

            # Check if stable attractor point
            delta = 1e-8
            dEqudx = (Equs(xss+delta, t, params)-Equs(xss, t, params))/delta
            jac = np.array([[dEqudx]])
            eig = jac
            instablility = np.real(eig) >= 0

            # Check if it is sufficiently large, has small residual, and successfully converges
            if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:
                # If so, it is a valid solution and we return it
                return xss

        # If no valid solutions are found after trying all initial guesses
        return float('nan')

    
# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(xss_list, alpha_list, n_list, choice1, choice2):
    
    # Evaluate sensitivities
    S_alpha_xss = S_alpha_xss_analytic(xss_list, alpha_list, n_list)
    S_n_xss     = S_n_xss_analytic(xss_list, alpha_list, n_list)

    # Sensitivity dictionary
    sensitivities = {
        "S_alpha_xss": S_alpha_xss,
        "S_n_xss": S_n_xss}

    # Map indices to keys
    labels = {
        0: "S_alpha_xss",
        1: "S_n_xss"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]


# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):

    '''
    Uncomment the following line if need to see solution printed.
    It will look like this:
    Solution:  {'alpha': 3.239629898497018, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.7701015749115143, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.6542032278143664, 'n': 9.910685695594527}
    Solution:  {'alpha': 2.6542032278143664, 'n': 9.363644921265}
    Solution:  {'alpha': 3.0278948409846858, 'n': 9.996303250351326}
    Solution:  {'alpha': 2.6451188083692183, 'n': 9.996303250351326}
    '''
    # print("Solution: ", solution)
	
	# Update parameter set
    alpha_val = solution["alpha"]
    n_val = solution["n"]

    # Create an empty numpy array
    xss_collect = np.array([])

    # Find steady states and store
    xss = ssfinder(alpha_val,n_val)
    xss_collect = np.append(xss_collect,xss)
    
    # Get sensitivity pair
    sens1, sens2 = senpair(xss_collect, solution["alpha"], solution["n"], choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    return ans1, ans2                                                                                #<--------- ...UP TO HERE



# ---------------------------------------------------------------------------- NEW STUFF ----------------------------------------------------------------------------



# _________________________________________________________________________________________

# For each run in our total number of runs
for run in range(1, runs + 1):

    # _________________________________________________________________________________________
    
    # There may be NaNs in the array. Pareto minimisation will think NaNs are minimum. We don't want this. Let's replace NaNs with infinities.
    Sens1 = np.where(np.isnan(Sens1), np.inf, Sens1)
    
    # Run Pareto tool with minimisation setting to get a mask.
    # Each mask is an array of the form [True, False, True, ...].
    # Indexing another array with this mask will remove rows corresponding to Falses.
    # Eg:  dummy       = [1   , 2   , 3    ]
    #      mask        = [True, True, False]
    #      dummy[mask] = [1   , 2          ]
    Sens1mask = paretoset(Sens1, sense=["min", "min"])
    paretoset_Sens1 = Sens1[Sens1mask]
    
    # Save tables of Pareto points for each sensitivity pair
    np.save('Paretos_Sens1' + str(aindex) + '.npy', paretoset_Sens1Pair1)
    
    
    print('checkpointE')
    
    # _________________________________________________________________________________________
    
    # Get the corresponding pareto fronts in parameter space
    paretoset_Params1  = ParamCombinations[Sens1mask]
    
    # Save the arrays
    np.save('Paretos_Params1' + str(aindex) + '.npy', paretoset_Params1)
    #_________________________________________________________________________________________
    
    ' + str(aindex) + '
    ' + str(aindex) + '
    
    print('checkpointF')
