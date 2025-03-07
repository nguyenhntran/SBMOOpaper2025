# -------------- PART 0: PYTHON PRELIM --------------

# Additional notes: 
# mosa.py evolve() function has been edited

# Import packages
import importlib
import os
import time
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy import random
import pandas as pd

# Name for text file to records stats
output_file = "OneSpecies.csv"


# -------------- PART 0a: CHOOSE CIRCUIT AND SET UP FOLDER --------------


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

if not os.path.exists('data'):
    os.makedirs('data')

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
    return ans1, ans2
    

# -------------- PART 1: GAUGING MOSA PARAMETERS --------------

# Sample alpha values
alpha_min = float(input("Please enter minimum alpha value: "))
alpha_max = float(input("Please enter maximum alpha value: "))
alpha_sampsize = int(input("Please enter the number of alpha samples: "))
alpha_samps = np.linspace(alpha_min, alpha_max, alpha_sampsize)

# Sample n values
n_min = float(input("Please enter minimum n value: "))
n_max = float(input("Please enter maximum n value: "))
n_sampsize = int(input("Please enter the number of n samples: "))
n_samps = np.linspace(n_min, n_max, n_sampsize)

# Create empty arrays to store corresponding values of xss and yss
xss_samps = np.array([])
sens1_samps = np.array([])
sens2_samps = np.array([])

# For each combination of parameters
for i in alpha_samps:
    for j in n_samps:
        
        # Get steady state value and store
        xss = ssfinder(i,j)
        xss_samps = np.append(xss_samps,xss)
        # Get corresponding sensitivities and store
        sens1, sens2 = senpair(xss, i, j, choice1, choice2)
        sens1_samps = np.append(sens1_samps,sens1)
        sens2_samps = np.append(sens2_samps,sens2)

# Get min and max of each sensitivity and print
sens1_samps_min = np.nanmin(sens1_samps)
sens2_samps_min = np.nanmin(sens2_samps)
sens1_samps_max = np.nanmax(sens1_samps)
sens2_samps_max = np.nanmax(sens2_samps)

# Get MOSA energies
deltaE_sens1 = sens1_samps_max - sens1_samps_min
deltaE_sens2 = sens2_samps_max - sens2_samps_min
deltaE = np.linalg.norm([deltaE_sens1, deltaE_sens2])

# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.9): "))
temp_hot = deltaE / np.log(1/probability_hot)

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.01): "))
temp_cold = deltaE / np.log(1/probability_cold)


# -------------- PART 2a: MOSA PREPARATIONS --------------


# Print prompts
print("Now preparing to MOSA...")
runs = int(input("Please enter number of MOSA runs you would like to complete (if in doubt enter 5): "))
iterations = int(input("Please enter number of random walks per run (if in doubt enter 100): "))

# Data record for each run
hotrun_time = []
hotrun_stoptemp = []
temp_num = []
step_scale = []
coldrun_time = []
coldrun_stoptemp = []
prune_time = []
archive_length_cold = []

# For each run
for run in range(runs):
    print(f"MOSA run number: {run+1}")
    
    # Define lists to collect sensitivity and parameter values from each MOSA run before pruning
    annealed_Salpha = []
    annealed_Sn     = []
    annealed_alpha  = []
    annealed_n      = []
    # Define lists to collect sensitivity and parameter values from each MOSA run after pruning
    pareto_Salpha = []
    pareto_Sn     = []
    pareto_alpha  = []
    pareto_n      = []
    
    # Delete archive and checkpoint json files at the start of each new run
    files_to_delete = ["archive.json", "checkpoint.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

	# -------------- PART 2b: ANNEAL TO GET PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Set random seed for MOSA
    random.seed(run)
	
	# Initialisation of MOSA
    opt = mosa.Anneal()
    opt.archive_size = 10000
    opt.maximum_archive_rejections = opt.archive_size
    opt.population = {"alpha": (alpha_min, alpha_max), "n": (n_min, n_max)}
	
	# Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = iterations
    opt.temperature_decrease_factor = 0.95
    opt.number_of_temperatures = int(np.ceil(np.log(temp_cold / temp_hot) / np.log(opt.temperature_decrease_factor)))
    opt.number_of_solution_elements = {"alpha":1, "n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
    	
    # Hot run
    start_time = time.time()
    hotrun_stoppingtemp = opt.evolve(fobj)
    
    hotrun_time.append(time.time() - start_time)
    hotrun_stoptemp.append(hotrun_stoppingtemp)
    temp_num.append(opt.number_of_temperatures)
    step_scale.append(step_scaling)
	
    # Cold run options
    opt.initial_temperature = hotrun_stoppingtemp
    opt.number_of_iterations = iterations
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"alpha":1,"n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"alpha": (alpha_max-alpha_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Cold run
    start_time = time.time()
    coldrun_stoppingtemp = opt.evolve(fobj)
    print(f"Cold run time: {time.time() - start_time} seconds")
    
    coldrun_time.append(time.time() - start_time)
    coldrun_stoptemp.append(coldrun_stoppingtemp)
    
    # Output 
    start_time = time.time()
    pruned = opt.prunedominated()
    
    prune_time.append(time.time() - start_time)
	
	# -------------- PART 2c: STORE AND PLOT UNPRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["alpha"] for solution in data["Solution"]])
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        annealed_Salpha.append(dummy1)
        annealed_Sn.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Unpruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'data/unpruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
	
    # -------------- PART 2d: STORE AND PLOT CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract alpha and n values from the solutions
    alpha_values = [solution["alpha"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(alpha_values, n_values):
        annealed_alpha.append(dummy1)
        annealed_n.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(alpha_values, n_values)
    plt.xlabel('alpha')
    plt.ylabel('n')
    plt.grid(True)
    plt.title(f'Unpruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'data/unpruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()

    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save S_a annealed values
    filename = f"annealed_Salpha_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_Salpha)
    # Save S_n annealed values
    filename = f"annealed_Sn_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_Sn)
    # Save a annealed values
    filename = f"annealed_alpha_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_alpha)
    # Save n annealed values
    filename = f"annealed_n_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_n)

    # -------------- PART 2f: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    data = pruned
        
    # Check archive length
    length = len([solution["alpha"] for solution in data["Solution"]])
    
    archive_length_cold.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_Salpha.append(dummy1)
        pareto_Sn.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'data/pruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
	
    # -------------- PART 2g: STORE AND PLOT CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract alpha and n values from the solutions
    alpha_values = [solution["alpha"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(alpha_values, n_values):
        pareto_alpha.append(dummy1)
        pareto_n.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(alpha_values, n_values)
    plt.xlabel('alpha')
    plt.ylabel('n')
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'data/pruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()

    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save S_a pareto values
    filename = f"pareto_Salpha_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_Salpha)
    # Save S_n pareto values
    filename = f"pareto_Sn_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_Sn)
    # Save a pareto values
    filename = f"pareto_alpha_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_alpha)
    # Save n pareto values
    filename = f"pareto_n_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_n)
    
# -------------- PART 2f: SAVE DATA -------------------------------------------

mosa_data = pd.DataFrame({
    'Circuit': [circuit],
    'Runs': [runs],
    'Random walks': [iterations],
    'Steady state': [numss],
    'alpha (min, max, samples)': [[alpha_min, alpha_max, alpha_sampsize]],
    'n (min, max, samples)': [[n_min, n_max, n_sampsize]],
    'S_alpha_xss sampled value (min, max)': [[sens1_samps_min, sens1_samps_max]],
    'S_n_xss sampled value (min, max)': [[sens2_samps_min, sens2_samps_max]],
    'S_alpha_xss sampled energy difference': [deltaE_sens1],
    'S_n_xss sampled energy difference': [deltaE_sens2],
    'Sampled cumulatve energy difference': [deltaE],
    'Probability hot': [probability_hot],
    'Hot run temperature': [temp_hot],
    'Probability cold': [probability_cold],
    'Cold run temperature': [temp_cold],
    'Hot run time': [hotrun_time],
    'Hot run stopping temperature': [hotrun_stoptemp],
    'Number of temperatures': [temp_num],
    'Step scaling factor': [step_scale],
    'Cold run time': [coldrun_time],
    'Cold run stopping temperature': [coldrun_stoptemp],
    'Prune time': [prune_time],
    'Archive length after cold run': [archive_length_cold]
})

file_exists = os.path.exists(output_file)
mosa_data.to_csv(output_file, mode='a', index=False, header=not file_exists)

print(f"Appended new data to {output_file}")

# -------------- PART 3a: VISUALISE EACH MOSA RECTANGULAR BOUNDS --------------

# Print prompt
print("Plotting rectangular bounds for each MOSA run...")

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, runs))

# Create plot
fig, ax = plt.subplots(figsize=(6, 6))

# For each run in our total number of runs
for run in range(1, runs + 1):

    # Load the parameter values for that run's Pareto front
    pareto_alpha = np.load(f"data/pareto_alpha_run{run}.npy", allow_pickle=True)
    pareto_n = np.load(f"data/pareto_n_run{run}.npy", allow_pickle=True)
    
    # Create 2D points from the loaded data
    points = np.array(list(zip(pareto_alpha, pareto_n)))
    
    # Compute the bounding box from the points
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # Define the bounding box (close the rectangle by repeating the first point)
    bounding_box = np.array([
        [min_x, min_y],  # Bottom-left
        [max_x, min_y],  # Bottom-right
        [max_x, max_y],  # Top-right
        [min_x, max_y],  # Top-left
        [min_x, min_y]   # Closing the rectangle
    ])
    
    # Pick a color for this run
    color = colors[run - 1]
    
    # Plot the bounding rectangle for this run with a unique color
    ax.plot(bounding_box[:, 0], bounding_box[:, 1],
            color=color, linewidth=2, label=f"MOSA Run {run}")

# Cosmetics
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$n$")
ax.set_title("Bounding Rectangle for Each MOSA Run")
ax.set_xlim([alpha_min, alpha_max])
ax.set_ylim([n_min, n_max])
ax.legend()

# Save the figure and close
plt.savefig('data/searchspaces_runs.png', dpi=300)
plt.close()
