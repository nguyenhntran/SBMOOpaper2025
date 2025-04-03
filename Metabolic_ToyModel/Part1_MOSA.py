# -------------- PART 0: PYTHON PRELIM --------------

# Additional notes: 
# mosa.py evolve() function has been edited to return final stopping temperature

# Import packages
import importlib
import os
import time
from tqdm import tqdm
import itertools
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from numpy import random
import pandas as pd


# -------------- PART 0: CHOOSE CIRCUIT AND SET UP FOLDER --------------


# Choose circuit
circuit = "ToyModel"

# Import circuit config file
config = importlib.import_module(circuit)

# Define the subfolder name
folder_name = f"MOSA_{circuit}"

# Create folder if not yet exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Jump to folder
os.chdir(folder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0b: DEFINE DYNAMICAL SYSTEM --------------

# Import ODEs
Equ1 = config.Equ1
Equ2 = config.Equ2
Equ3 = config.Equ3
Equ4 = config.Equ4
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x0 = P[0]
    x1 = P[1]
    e1 = P[2]
    e2 = P[3]
    k1 = params[0]
    k2 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    val0 = Equ1(x0, e1)
    val1 = Equ2(x0, x1, e1, e2)
    val2 = Equ3(x1, e1, k1, theta1)
    val3 = Equ4(x1, e2, k2, theta2)
    return np.array([val0, val1, val2, val3])

# Define initial time
t = 0.0


# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------


# Define analytical sensitivity expressions
S_k1_x1_analytic = config.S_k1_x1_analytic
S_k2_x1_analytic = config.S_k2_x1_analytic


# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------


# Print prompt
print("""
We have the following sensitivity functions:
0. |S_k1_x1|
1. |S_k2_x1|
2. |S_theta1_x1|
3. |S_theta2_x1|

Only 0 and 1 are available now.
""")

# Choose pair of functions
choice1 = int(0)
choice2 = int(1)

# List of sensitivity function names
sensitivity_labels = [
    "|S_k1_x1|",
    "|S_k2_x1|",
    "|S_theta1_x1|",
    "|S_theta2_x1|"]

# Save function names for later use
label1 = sensitivity_labels[choice1]
label2 = sensitivity_labels[choice2]

# Name for text file to records stats
output_file = f"Metabolic1_{choice1}_and_{choice2}.csv"


# -------------- PART 0e: CHANGING DIRECTORIES --------------


# Define the subfolder name
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


# DEFINE FUNCTION THAT SOLVES FOR STEADY STATE OBSERVABLES TO BE MINIMISED BY MOSA

def ssfinder(k1_val, k2_val, theta1_val, theta2_val):
    # Define initial guesses (assumes a config function exists that gives them)
    InitGuesses = config.generate_initial_guesses()
    
    # Define parameter array
    params = np.array([k1_val, k2_val, theta1_val, theta2_val])
    
#    print("params: ", params)
    
    # Loop through all guesses
    for InitGuess in InitGuesses:
        # Solve for steady state
        output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(0, params), xtol=1e-12, full_output=True)
        
#        print("InitGuess: ", InitGuess)
#        print("output: ", output)
        
        x0ss, x1ss, e1ss, e2ss = output
        fvec = infodict['fvec'] #divergence
        
        # Check Jacobian for stability
        delta = 1e-8
        J = np.zeros((4, 4))
        for i in range(4):
            dP = np.zeros(4)
            dP[i] = delta
            J[:, i] = (Equs(output + dP, 0, params) - Equs(output, 0, params)) / delta
        
        eigvals = np.linalg.eigvals(J)
#        print("Eigenvalues")
#        print(eigvals)
        unstable = np.any(np.real(eigvals) >= 0)
        
        # Check if steady state is valid
        if np.all(output >= 0) and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and not unstable:
            return x0ss, x1ss, e1ss, e2ss

    # Return NaNs if no valid steady state is found
    return float('nan'), float('nan'), float('nan'), float('nan')
    
        
# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(x0ss_val, x1ss_val, e1ss_val, e2ss_val, k1_val, k2_val, theta1_val, theta2_val, choice1, choice2):
    
    # Evaluate sensitivities
    S_k1_x1 = S_k1_x1_analytic(x0ss_val, x1ss_val, e1ss_val, e2ss_val, k1_val, k2_val, theta1_val, theta2_val)
    S_k2_x1 = S_k2_x1_analytic(x0ss_val, x1ss_val, e1ss_val, e2ss_val, k1_val, k2_val, theta1_val, theta2_val)

    # Sensitivity dictionary
    sensitivities = {
        "S_k1_x1": S_k1_x1,
        "S_k2_x1": S_k2_x1}

    # Map indices to keys
    labels = {
        0: "S_k1_x1",
        1: "S_k2_x1"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]

# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):
	
	# Update parameter set
    k1_val = 10**(solution["k1"])
    k2_val = 10**(solution["k2"])
    theta1_val = 10**(solution["theta1"])
    theta2_val = 10**(solution["theta2"])
    
    # Check parameters
    print("params", [k1_val, k2_val, theta1_val, theta2_val])

    # Find steady states and store.
    x0ss_val, x1ss_val, e1ss_val, e2ss_val = ssfinder(k1_val, k2_val, theta1_val, theta2_val)
    if np.isnan(x0ss_val) or np.isnan(x1ss_val) or np.isnan(e1ss_val) or np.isnan(e2ss_val):
        return np.inf, np.inf, np.inf
    
    # Get sensitivity pair
    sens1, sens2 = senpair(x0ss_val, x1ss_val, e1ss_val, e2ss_val, k1_val, k2_val, theta1_val, theta2_val, choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    
    
    #-------------------- J -----------------------

    # Constants
    V_in = 1
    k_cat = 12
    k_m = 10
    
    # Initial conditions
    y0 = [2290,0,0,0]
    params = [k1_val, k2_val, theta1_val, theta2_val]
    t_max = 5e4 
    t_eval = np.linspace(0, t_max)
    
    # Run the integration
    sol = solve_ivp(
        fun=lambda t, y: Equs(y, t, params),
        t_span=[0, t_max],
        y0=y0,
        method='RK23',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True)
    
    t_used = sol.t
    x1_vals = sol.y[1]  # x1(t)
    e2_vals = sol.y[3]  # e2(t)

    # Compute integrand at each time point
    integrand_vals = np.abs(V_in - e2_vals * (k_cat * x1_vals) / (k_m + x1_vals))
    
    # Compute the integral numerically using trap rule
    integral_val = np.trapz(integrand_vals, t_used)

    integral_val_rescaled = integral_val / 10000
    
    print(f"Integral up to steady state: {integral_val_rescaled:.6f}")          

    
    #-------------------------------------------
    
    
    # Check for NaN values before returning
    if np.isnan(ans1) or np.isnan(ans2):
        return np.inf, np.inf, np.inf
    return ans1, ans2, integral_val_rescaled



# -------------- PART 1: GAUGING MOSA PARAMETERS --------------

    
# Sample rS1 values
k1_min = 0.0000001
k1_max = 0.001
k1_sampsize = 6
k1_samps = np.linspace(k1_min, k1_max, k1_sampsize)

# Sample kS1 values
k2_min = 0.0000001
k2_max = 0.001
k2_sampsize = 6
k2_samps = np.linspace(k2_min, k2_max, k2_sampsize)

# Sample kS2 values
theta1_min = 0.001
theta1_max = 10
theta1_sampsize = 6
theta1_samps = np.linspace(theta1_min, theta1_max, theta1_sampsize)

# Sample kP values
theta2_min = 0.001
theta2_max = 10
theta2_sampsize = 6
theta2_samps = np.linspace(theta2_min, theta2_max, theta2_sampsize)


# Create empty arrays to store ...
# ... steady states
x0ss_samps = np.array([])
x1ss_samps = np.array([])
# ... sensitivities
sens1_samps = np.array([])
sens2_samps = np.array([])



# WITH LOADING BAR 
# Compute the total number of iterations for tqdm
total_iterations = k1_sampsize * k2_sampsize * theta1_sampsize * theta2_sampsize
# Loop over every combination of parameters with a progress bar
for i, j, k, l in tqdm(itertools.product(k1_samps, k2_samps, theta1_samps, theta2_samps), total=total_iterations, desc="Gauging energies:"):
    
    # Get steady states and store
    x0ss, x1ss, e1ss, e2ss = ssfinder(i, j, k, l)
    
    x0ss_samps = np.append(x0ss_samps, x0ss)
    x1ss_samps = np.append(x1ss_samps, x1ss)

    # Get sensitivities and store
    sens1, sens2 = senpair(x0ss, x1ss, e1ss, e2ss, i, j, k, l, choice1, choice2)
    sens1_samps = np.append(sens1_samps, sens1)
    sens2_samps = np.append(sens2_samps, sens2)

# Get min and max of each sensitivity and print
sens1_samps_min = np.nanmin(sens1_samps)
sens2_samps_min = np.nanmin(sens2_samps)
sens1_samps_max = np.nanmax(sens1_samps)
sens2_samps_max = np.nanmax(sens2_samps)

# Check values
print("Sensitivity ranges: ", sens1_samps_min, sens2_samps_min, sens1_samps_max, sens2_samps_max)

# Get MOSA energies
deltaE_sens1 = abs(sens1_samps_max - sens1_samps_min)
deltaE_sens2 = abs(sens2_samps_max - sens2_samps_min)
deltaE = np.linalg.norm([deltaE_sens1, deltaE_sens2])
print("deltaE: ", deltaE)


# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(0.9)
temp_hot = deltaE / np.log(1/probability_hot) #50 
print("temp_hot")
print(temp_hot)

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(0.01)
temp_cold = deltaE / np.log(1/probability_cold)
print("temp_cold")
print(temp_cold)



# -------------- PART 2a: PREPPING MOSA --------------


# Print prompts
print("Now preparing to MOSA...")
runs = int(input("Please enter number of MOSA runs you would like to complete (if in doubt enter 5): "))
iterations = int(input("Please enter number of random walks per run (if in doubt enter 100): "))

hotrun_time = []
hotrun_stoptemp = []
temp_num = []
step_scale_hot = []
step_scale_cold = []
coldrun_time = []
coldrun_stoptemp = []
prune_time = []
archive_cold_len = []
archive_prune_len = []

# For each run
for run in range(runs):
    print(f"MOSA run number: {run+1}")
    
    # Define lists to collect sensitivity and parameter values from each MOSA run before pruning
    annealed_sensfunc1 = []
    annealed_sensfunc2 = []
    annealed_J = []
    annealed_k1        = []
    annealed_k2        = []
    annealed_theta1    = []
    annealed_theta2    = []
    
    # Define lists to collect sensitivity and parameter values from each MOSA run after pruning
    pareto_sensfunc1 = []
    pareto_sensfunc2 = []
    pareto_J = []
    pareto_k1        = []
    pareto_k2        = []
    pareto_theta1    = []
    pareto_theta2    = []
    
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
    opt.population = {"k1": (np.log10(k1_min), np.log10(k1_max)),"k2": (np.log10(k2_min), np.log10(k2_max)),"theta1": (np.log10(theta1_min), np.log10(theta1_max)), "theta2": (np.log10(theta2_min), np.log10(theta2_max))}
    
    # Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = iterations
    opt.temperature_decrease_factor = 0.95
    opt.number_of_temperatures = int(np.ceil(np.log(temp_cold / temp_hot) / np.log(opt.temperature_decrease_factor)))
    opt.number_of_solution_elements = {"k1": 1,"k2": 1,"theta1": 1, "theta2": 1}
    scale = 0.7
    opt.mc_step_size = {"k1": abs(np.log10(k1_min) - np.log10(k1_max))*scale,"k2": abs(np.log10(k2_min) - np.log10(k2_max))*scale,"theta1": abs(np.log10(theta1_min) - np.log10(theta1_max))*scale, "theta2": abs(np.log10(theta2_min) - np.log10(theta2_max))*scale}
	
	# Hot run
    start_time = time.time()
    hotrun_stoppingtemp = opt.evolve(fobj)
    
    # Record stats
    hotrun_time.append(time.time() - start_time)
    hotrun_stoptemp.append(hotrun_stoppingtemp)
    temp_num.append(opt.number_of_temperatures)
    step_scale_hot.append(scale)
    
    # Cold run options
    opt.initial_temperature = hotrun_stoppingtemp
    opt.number_of_iterations = iterations
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.93
    opt.number_of_solution_elements = {"rS1": 1,"rS2": 1,"rS3": 1, "kS1": 1, "kS2": 1, "kP": 1, "kE1": 1, "kE2": 1, "alphaE": 1, "gammaE": 1}
    scale = 0.3
    opt.mc_step_size = {"k1": abs(np.log10(k1_min) - np.log10(k1_max))*scale,"k2": abs(np.log10(k2_min) - np.log10(k2_max))*scale,"theta1": abs(np.log10(theta1_min) - np.log10(theta1_max))*scale, "theta2": abs(np.log10(theta2_min) - np.log10(theta2_max))*scale}
	
    # Cold run
    start_time = time.time()
    coldrun_stoppingtemp = opt.evolve(fobj)
    
    # Record stats
    coldrun_time.append(time.time() - start_time)
    coldrun_stoptemp.append(coldrun_stoppingtemp)
    step_scale_cold.append(scale)
    
    # Pruning 
    start_time = time.time()
    pruned = opt.prunedominated()
    
    # Record stats 
    prune_time.append(time.time() - start_time)
    
    # -------------- PART 2c: STORE AND PLOT UNPRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["k1"] for solution in data["Solution"]])
    
    # Record stats 
    archive_cold_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    value_3 = [v[2] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3 in zip(value_1, value_2, value_3):
        annealed_sensfunc1.append(dummy1)
        annealed_sensfunc2.append(dummy2)
        annealed_J.append(dummy3)
    
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
    
    # Extract parameter values from the solutions
    k1_values = [solution["k1"] for solution in data["Solution"]]
    k2_values = [solution["k2"] for solution in data["Solution"]]
    theta1_values = [solution["theta1"] for solution in data["Solution"]]
    theta2_values = [solution["theta2"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3, dummy4 in zip(k1_values, k2_values, theta1_values, theta2_values):
        annealed_k1.append(dummy1)
        annealed_k2.append(dummy2)
        annealed_theta1.append(dummy3)
        annealed_theta2.append(dummy4)
        
    # Create a figure with 4 1D scatter plots
    
    params = {
        "k1": annealed_k1,
        "k2": annealed_k2,
        "theta1": annealed_theta1,
        "theta2": annealed_theta2}
    
    fig, axes = plt.subplots(len(params), 1, figsize=(8, 12), sharex=False)
    
    for i, (param_name, values) in enumerate(params.items()):
        axes[i].scatter(values, [i] * len(values), alpha=0.6)
        axes[i].set_ylabel(param_name)
        axes[i].set_yticks([i])
        axes[i].set_yticklabels([param_name])
    
    plt.xlabel("Parameter Values")
    plt.title(f'Unpruned MOSA Pareto Parameters - Run No. {run + 1}')
    plt.savefig(f'data/unpruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.tight_layout()
    plt.close()
    
    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save sensitivity function 1 annealed values
    filename = f"annealed_sensfunc1_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_sensfunc1)
    # Save sensitivity function 2 annealed values
    filename = f"annealed_sensfunc2_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_sensfunc2)
    # Save J annealed values
    filename = f"annealed_J_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_J)
    # Save k1 annealed values
    filename = f"annealed_k1_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_k1)
    # Save k2 annealed values
    filename = f"annealed_k2_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_k2)
    # Save theta1 annealed values
    filename = f"annealed_theta1_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_theta1)
    # Save theta2 annealed values
    filename = f"annealed_theta2_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_theta2)
    
    # -------------- PART 2f: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    data = pruned
        
    # Check archive length
    length = len([solution["k1"] for solution in data["Solution"]])

    # Record stats
    archive_prune_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    value_3 = [v[2] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3 in zip(value_1, value_2, value_3):
        pareto_sensfunc1.append(dummy1)
        pareto_sensfunc2.append(dummy2)
        pareto_J.append(dummy3)

    
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
    
    # Extract parameter values from the solutions
    k1_values = [solution["k1"] for solution in data["Solution"]]
    k2_values = [solution["k2"] for solution in data["Solution"]]
    theta1_values = [solution["theta1"] for solution in data["Solution"]]
    theta2_values = [solution["theta2"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3, dummy4 in zip(k1_values, k2_values, theta1_values, theta2_values):
        pareto_k1.append(dummy1)
        pareto_k2.append(dummy2)
        pareto_theta1.append(dummy3)
        pareto_theta2.append(dummy4)
        
    # Create a figure with 4 1D scatter plots
    params = {
        "k1": pareto_k1,
        "k2": pareto_k2,
        "theta1": pareto_theta1,
        "theta2": pareto_theta2}
    
    fig, axes = plt.subplots(len(params), 1, figsize=(8, 12), sharex=False)
    
    for i, (param_name, values) in enumerate(params.items()):
        axes[i].scatter(values, [i] * len(values), alpha=0.6)
        axes[i].set_ylabel(param_name)
        axes[i].set_yticks([i])
        axes[i].set_yticklabels([param_name])
    
    plt.xlabel("Parameter Values")
    plt.title(f'Pruned MOSA Pareto Parameters - Run No. {run + 1}')
    fig.savefig(f'data/pruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.tight_layout()
    plt.close()
    
    # -------------- PART 2h: SAVE PARETO DATA FROM CURRENT RUN --------------
    
    # Save sensitivity function 1 pareto values
    filename = f"pareto_sensfunc1_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_sensfunc1)
    # Save sensitivity function 2 pareto values
    filename = f"pareto_sensfunc2_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_sensfunc2)
    # Save J pareto values
    filename = f"pareto_J_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_J)
    # Save k1 pareto values
    filename = f"pareto_k1_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_k1)
    # Save k2 pareto values
    filename = f"pareto_k2_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_k2)
    # Save theta1 pareto values
    filename = f"pareto_theta1_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_theta1)
    # Save theta2 pareto values
    filename = f"pareto_theta2_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_theta2)
    
    # ---------------- PART 2i: SAVE MOSA DATA --------------------------------
    
    mosa_data = pd.DataFrame({
        'Circuit': [circuit],
        'Runs': [runs],
        'Random walks per run': [iterations],
        'k1 (min, max, samples)': [[k1_min, k1_max, k1_sampsize]],
        'k2 (min, max, samples)': [[k2_min, k2_max, k2_sampsize]],
        'theta1 (min, max, samples)': [[theta1_min, theta1_max, theta1_sampsize]],
        'theta2 (min, max, samples)': [[theta2_min, theta2_max, theta2_sampsize]],
        f'(min, max) sampled value of {label1}': [[sens1_samps_min, sens1_samps_max]],
        f'(min, max) sampled value of {label2}': [[sens2_samps_min, sens2_samps_max]],
        f'Sampled energy difference in {label1}': [deltaE_sens1],
        f'Sampled energy difference in {label2}': [deltaE_sens2],
        'Sampled cumulative energ difference': [deltaE],
        'Probability hot run': [probability_hot],
        'Hot run temperature': [temp_hot],
        'Probability cold run': [probability_cold],
        'Cold run temperature': [temp_cold],
        'Hot run time': [hotrun_time],
        'Hot run stopping temperature': [hotrun_stoptemp],
        'Number of temperatures': [temp_num],
        'Step scaling factor in hot run': [step_scale_hot],
        'Step scaling factor in cold run': [step_scale_cold],
        'Cold run time': [coldrun_time],
        'Cold run stopping temperature': [coldrun_stoptemp],
        'Prune time': [prune_time],
        'Archive length after cold run': [archive_cold_len],
        'Archive length after prune': [archive_prune_len]
    })

file_exists = os.path.exists(output_file)
mosa_data.to_csv(output_file, mode='a', index=False, header=not file_exists)

print(f"Appended new data to {output_file}")
