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


# Load data
grid_runevolving = np.load(f"grid_runevolving.npy", allow_pickle=True)

# Unpack parameters
a_vals = grid_new[:, 0]
n_vals = grid_new[:, 1]
run_vals = grid_new[:, 2]

# We are only taking one alpha value at a time
aindex = int(sys.argv[1])
a_vals = np.array([a_vals[aindex]])

# Number of a and n values
a_no = 1
n_no = len(n_vals)

# _________________________________________________________________________________________

#  We want the following array
#                        --------------------------------------------
#                       |    a value   |    n value   |   run value  |
#                       |       #      |       #      |       #      | <--- row 0
#  ParamCombinations =  |       #      |       #      |       #      | 
#                       |       #      |       #      |       #      | 
#                       |       #      |       #      |       #      | <--- row n_no - 1
#                        --------------------------------------------

# Initialise memory
ParamCombinations = np.full( (a_no*n_no , 3) , None )

# Dummy counter to track current row in table
currentrow = 0

# For each coordinate in parameter space given by the row in our table
for a_val in a_vals:
  for n_val, run_val in zip(n_vals, run_vals):
    ParamCombinations[currentrow,:] = np.array([a_val, n_val, run_val])
    # Update current row
    currentrow += 1

# Save data
np.save('ParamCombinations' + str(aindex) + '.npy', ParamCombinations)


print("CheckpointB")


# _________________________________________________________________________________________

#  We want the following array
#            ------------------------
#           |   xss   |   run value  |  
#           |    #    |       #      | <--- row 0 
#  xss1  =  |    #    |       #      |
#           |    #    |       #      |  
#           |    #    |       #      | <--- row n_no - 1
#            ------------------------

# Get number of rows in table
rows = ParamCombinations.shape[0]

# Initialize empty array to store steady-state values
xss1 = np.empty((rows, 2), dtype=float)

# For each (a,n) point in our search space given by rows of ParamCombinations array
for row in range(rows):

    # Extract the parameter values (a_val and n_val) from the current row
    a_val = float(ParamCombinations[row, 0])
    n_val = float(ParamCombinations[row, 1])
    run_val = int(ParamCombinations[row, 2])
    
    # Get steady state value
    xss = ssfinder(a_val,n_val)
    
    # Record value
    xss1[row, 0] = xss[0]
    xss1[row, 1] = run_val

# Save data
np.save('steadystates' + str(aindex) + '.npy', xss1)

print('checkpointC')



# _________________________________________________________________________________________

#  We want the following array
#             --------------------------------------------------
#            |    S_{a}(xss)   |    S_{n}(yss)   |   run value  |
#            |         #       |         #       |       #      | <--- row 0 
#  Sens1  =  |         #       |         #       |       #      |
#            |         #       |         #       |       #      |
#            |         #       |         #       |       #      | <--- row n_no - 1
#             --------------------------------------------------

# Initialize empty array to store sensitivity pair values of interst
Sens1 = np.empty([rows, 3])

# For each row in parameter table
for row in range(rows):

    # get the corresponding parameter values
    a_val = float(ParamCombinations[row, 0])  # Ensure numerical type
    n_val = float(ParamCombinations[row, 1])  # Ensure numerical type
    run_val = int(ParamCombinations[row, 2])

    # get the corresponding steady state 1 values
    xss1_val = float(xss1[row, 0])

    # get sensitivity values
    S_a_xss_val1, S_n_xss_val1 = senpair(xss1_val, a_val, n_val, choice1, choice2)

    # Add sensitivity values to sensitivity table 1
    Sens1[row,:] = np.array([S_a_xss_val1, S_n_xss_val1, run_val])

# Save data
np.save('SensitivityPair' + str(aindex) + '.npy', Sens1)

print('checkpointD')

# _________________________________________________________________________________________


' + str(aindex) + '
' + str(aindex) + '

print('checkpointE')
