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
from numpy import random
import pandas as pd


# -------------- PART 0: CHOOSE CIRCUIT AND SET UP FOLDER --------------


# Choose circuit
circuit = "metabolic1"

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


# Define number of steady states expected
numss = int(1)


# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------


# Define analytical sensitivity expressions
S_kE1_P_analytic = config.S_kE1_P_analytic
S_kE2_P_analytic = config.S_kE2_P_analytic
S_kE1_S2_analytic = config.S_kE1_S2_analytic
S_kE2_S2_analytic = config.S_kE2_S2_analytic
S_kE1_ATP_analytic = config.S_kE1_ATP_analytic
S_kE2_ATP_analytic = config.S_kE2_ATP_analytic


# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------


# Print prompt
print("""
We have the following sensitivity functions:
0. |S_kE1_P|
1. |S_kE2_P|
2. |S_kE1_S2|
3. |S_kE2_S2|
4. |S_kE1_ATP|
5. |S_kE2_ATP|

To MOSa for Pareto front of:
Sensitivity of Product Yield to Reaction Rates k_{E1}, k_{E2} --- choose 0 and 1.
Sensitivity of Intermediate Metabolite S2 to Reaction Rates k_{E1}, k_{E2}  --- choose 2 and 3.
Sensitivity of ATP Consumption to Reaction Rates k_{E1}, k_{E2}  --- choose 4 and 5.
""")

# Choose pair of functions
choice1 = int(input("Please select first option number:"))
choice2 = int(input("Please select second option number:"))

# List of sensitivity function names
sensitivity_labels = [
    "|S_kE1_P|",
    "|S_kE2_P|",
    "|S_kE1_S2|",
    "|S_kE2_S2|",
    "|S_kE1_ATP|",
    "|S_kE2_ATP|"]

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
def ssfinder(rS1_val, kS1_val, kS2_val, kP_val, kE1_val, kE2_val, alphaE_val, gammaE_val):

    # If we have one steady state
    if numss == 1: 

        S2ss = (1/kS2_val) * kE1_val * (kS1_val/rS1_val) * (alphaE_val/gammaE_val)
        S2ss_prime = S2ss
        Pss =  (1/kP_val) * kE1_val * (kE2_val/kS2_val) * (kS1_val/rS1_val) * (alphaE_val/gammaE_val)**2
        Pss_prime = - Pss
        ATPss = (alphaE_val/gammaE_val)*(kS1_val/rS1_val) * (kE1_val + kE2_val*(kE1_val/kS2_val)*(alphaE_val/gammaE_val))
        ATPss_prime = ATPss
        Ess = 2 * alphaE_val/gammaE_val
        Ess_prime = Ess
        
        return S2ss_prime, Pss_prime, ATPss_prime, Ess_prime
        
# DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
def senpair(rS1_val, kS1_val, kS2_val, kP_val, kE1_val, kE2_val, alphaE_val, gammaE_val, choice1, choice2):
    
    # Evaluate sensitivities
    S_kE1_P   = S_kE1_P_analytic()
    S_kE2_P   = S_kE2_P_analytic()
    S_kE1_S2  = S_kE1_S2_analytic()
    S_kE2_S2  = S_kE2_S2_analytic()
    S_kE1_ATP = S_kE1_ATP_analytic()
    S_kE2_ATP = S_kE2_ATP_analytic(kS2_val, kE2_val, alphaE_val, gammaE_val)

    # Sensitivity dictionary
    sensitivities = {
        "S_kE1_P": S_kE1_P,
        "S_kE2_P": S_kE2_P,
        "S_kE1_S2": S_kE1_S2,
        "S_kE2_S2": S_kE2_S2,
        "S_kE1_ATP": S_kE1_ATP,
        "S_kE2_ATP": S_kE2_ATP}

    # Map indices to keys
    labels = {
        0: "S_kE1_P",
        1: "S_kE2_P",
        2: "S_kE1_S2",
        3: "S_kE2_S2",
        4: "S_kE1_ATP",
        5: "S_kE2_ATP"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]

# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):
	
	# Update parameter set
    rS1_val = solution["rS1"]
    kS1_val = solution["kS1"]
    kS2_val = solution["kS2"]
    kP_val = solution["kP"]
    kE1_val = solution["kE1"]
    kE2_val = solution["kE2"]
    alphaE_val = solution["alphaE"]
    gammaE_val = solution["gammaE"]

    # Find steady states and store.
    S2ss_prime, Pss_prime, ATPss_prime, Ess_prime = ssfinder(rS1_val, kS1_val, kS2_val, kP_val, kE1_val, kE2_val, alphaE_val, gammaE_val)

    # Get sensitivity pair
    sens1, sens2 = senpair(rS1_val, kS1_val, kS2_val, kP_val, kE1_val, kE2_val, alphaE_val, gammaE_val, choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    
    # Return the quantities to be minimised by MOSA
    return S2ss_prime, Pss_prime, ATPss_prime, Ess_prime, ans1, ans2


# -------------- PART 1: GAUGING MOSA PARAMETERS --------------

    
# Sample rS1 values
rS1_min = 0.01
rS1_max = 5
rS1_sampsize = 4
rS1_samps = np.linspace(rS1_min, rS1_max, rS1_sampsize)

# Sample kS1 values
kS1_min = 0.01
kS1_max = 5
kS1_sampsize = 4
kS1_samps = np.linspace(kS1_min, kS1_max, kS1_sampsize)

# Sample kS2 values
kS2_min = 0.01
kS2_max = 5
kS2_sampsize = 4
kS2_samps = np.linspace(kS2_min, kS2_max, kS2_sampsize)

# Sample kP values
kP_min = 0.01
kP_max = 5
kP_sampsize = 4
kP_samps = np.linspace(kP_min, kP_max, kP_sampsize)

# Sample kE1 values
kE1_min = 0.01
kE1_max = 5
kE1_sampsize = 4
kE1_samps = np.linspace(kE1_min, kE1_max, kE1_sampsize)

# Sample kE2 values
kE2_min = 0.01
kE2_max = 5
kE2_sampsize = 4
kE2_samps = np.linspace(kE2_min, kE2_max, kE2_sampsize)

# Sample alphaE values
alphaE_min = 0.01
alphaE_max = 5
alphaE_sampsize = 4
alphaE_samps = np.linspace(alphaE_min, alphaE_max, alphaE_sampsize)

# Sample gammaE values
gammaE_min = 0.01
gammaE_max = 5
gammaE_sampsize = 4
gammaE_samps = np.linspace(gammaE_min, gammaE_max, gammaE_sampsize)


# Create empty arrays to store ...
# ... observables
S2ss_prime_samps = np.array([])
Pss_prime_samps = np.array([])
ATPss_prime_samps = np.array([])
Ess_prime_samps = np.array([])
# ... sensitivities
sens1_samps = np.array([])
sens2_samps = np.array([])




# WITH LOADING BAR 
# Compute the total number of iterations for tqdm
total_iterations = rS1_sampsize * kS1_sampsize * kS2_sampsize * kP_sampsize * kE1_sampsize * kE2_sampsize * alphaE_sampsize * gammaE_sampsize
# Loop over every combination of parameters with a progress bar
for i, j, k, l, m, n, o, p in tqdm(itertools.product(rS1_samps, kS1_samps, kS2_samps, kP_samps, kE1_samps, kE2_samps, alphaE_samps, gammaE_samps), total=total_iterations, desc="Gauging energies:"):
    
    # Get steady states and store
        S2ss_prime, Pss_prime, ATPss_prime, Ess_prime = ssfinder(i, j, k, l, m, n, o, p)
        
        S2ss_prime_samps = np.append(S2ss_prime_samps, S2ss_prime)
        Pss_prime_samps = np.append(Pss_prime_samps, Pss_prime)
        ATPss_prime_samps = np.append(ATPss_prime_samps, ATPss_prime)
        Ess_prime_samps = np.append(Ess_prime_samps, Ess_prime)

        # Get sensitivities and store
        sens1, sens2 = senpair(i, j, k, l, m, n, o, p, choice1, choice2)
        sens1_samps = np.append(sens1_samps, sens1)
        sens2_samps = np.append(sens2_samps, sens2)

# Get min and max of each sensitivity and print

S2ss_prime_samps_min = np.nanmin(S2ss_prime_samps)
Pss_prime_samps_min = np.nanmin(Pss_prime_samps)
ATPss_prime_samps_min = np.nanmin(ATPss_prime_samps)
Ess_prime_samps_min = np.nanmin(Ess_prime_samps)
sens1_samps_min = np.nanmin(sens1_samps)
sens2_samps_min = np.nanmin(sens2_samps)

S2ss_prime_samps_max = np.nanmax(S2ss_prime_samps)
Pss_prime_samps_max = np.nanmax(Pss_prime_samps)
ATPss_prime_samps_max = np.nanmax(ATPss_prime_samps)
Ess_prime_samps_max = np.nanmax(Ess_prime_samps)
sens1_samps_max = np.nanmax(sens1_samps)
sens2_samps_max = np.nanmax(sens2_samps)

# Get MOSA energies

deltaE_S2ss_prime = abs(S2ss_prime_samps_max - S2ss_prime_samps_min)
deltaE_Pss_prime = abs(Pss_prime_samps_max - Pss_prime_samps_min)
deltaE_ATPss_prime = abs(ATPss_prime_samps_max - ATPss_prime_samps_min)
deltaE_Ess_prime = abs(Ess_prime_samps_max - Ess_prime_samps_min)

deltaE_sens1 = abs(sens1_samps_max - sens1_samps_min)
deltaE_sens2 = abs(sens2_samps_max - sens2_samps_min)

deltaE = np.linalg.norm([deltaE_S2ss_prime, deltaE_Pss_prime, deltaE_ATPss_prime, deltaE_Ess_prime, deltaE_sens1, deltaE_sens2])


# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(0.9)
temp_hot = deltaE / np.log(1/probability_hot)

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(0.01)
temp_cold = deltaE / np.log(1/probability_cold)


# -------------- PART 2a: PREPPING MOSA --------------


# Print prompts
print("Now preparing to MOSA...")
runs = int(2)
iterations = int(100)

hotrun_time = []
hotrun_stoptemp = []
temp_num = []
step_scale = []
coldrun_time = []
coldrun_stoptemp = []
prune_time = []
archive_cold_len = []
archive_prune_len = []

# For each run
for run in range(runs):
    print(f"MOSA run number: {run+1}")
    
    # Define lists to collect sensitivity and parameter values from each MOSA run before pruning
    
    annealed_S2ss_prime  = []
    annealed_Pss_prime   = []
    annealed_ATPss_prime = []
    annealed_Ess_prime = []
    annealed_sensfunc1   = []
    annealed_sensfunc2   = []
    
    annealed_rS1    = []
    annealed_kS1    = []
    annealed_kS2    = []
    annealed_kP     = []
    annealed_kE1    = []
    annealed_kE2    = []
    annealed_alphaE = []
    annealed_gammaE = []
    
    # Define lists to collect sensitivity and parameter values from each MOSA run after pruning
    
    pareto_S2ss_prime  = []
    pareto_Pss_prime   = []
    pareto_ATPss_prime = []
    pareto_Ess_prime = []
    pareto_sensfunc1   = []
    pareto_sensfunc2   = []
    
    pareto_rS1    = []
    pareto_kS1    = []
    pareto_kS2    = []
    pareto_kP     = []
    pareto_kE1    = []
    pareto_kE2    = []
    pareto_alphaE = []
    pareto_gammaE = []
    
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
    opt.population = {"rS1": (rS1_min, rS1_max),"rS2": (rS1_min, rS1_max),"rS3": (rS1_min, rS1_max), "kS1": (kS1_min, kS1_max), "kS2": (kS2_min, kS2_max), "kP": (kP_min, kP_max), "kE1": (kE1_min, kE1_max), "kE2": (kE2_min, kE2_max), "alphaE": (alphaE_min, alphaE_max), "gammaE": (gammaE_min, gammaE_max)}
    
    # Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = iterations
    opt.temperature_decrease_factor = 0.95
    opt.number_of_temperatures = int(np.ceil(np.log(temp_cold / temp_hot) / np.log(opt.temperature_decrease_factor)))
    opt.number_of_solution_elements = {"rS1": 1,"rS2": 1,"rS3": 1, "kS1": 1, "kS2": 1, "kP": 1, "kE1": 1, "kE2": 1, "alphaE": 1, "gammaE": 1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size = {"rS1": abs(rS1_min - rS1_max)*step_scaling,"rS2": abs(rS1_min - rS1_max)*step_scaling,"rS3": abs(rS1_min - rS1_max)*step_scaling, "kS1": abs(kS1_min - kS1_max)*step_scaling, "kS2": abs(kS2_min - kS2_max)*step_scaling, "kP": abs(kP_min - kP_max)*step_scaling, "kE1": abs(kE1_min - kE1_max)*step_scaling, "kE2": abs(kE2_min - kE2_max)*step_scaling, "alphaE": abs(alphaE_min - alphaE_max)*step_scaling, "gammaE": abs(gammaE_min - gammaE_max)*step_scaling}
	
	# Hot run
    start_time = time.time()
    hotrun_stoppingtemp = opt.evolve(fobj)
    
    # Record stats
    hotrun_time.append(time.time() - start_time)
    hotrun_stoptemp.append(hotrun_stoppingtemp)
    temp_num.append(opt.number_of_temperatures)
    step_scale.append(step_scaling)
    
    # Cold run options
    opt.initial_temperature = hotrun_stoppingtemp
    opt.number_of_iterations = iterations
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"rS1": 1,"rS2": 1,"rS3": 1, "kS1": 1, "kS2": 1, "kP": 1, "kE1": 1, "kE2": 1, "alphaE": 1, "gammaE": 1}
    step_scaling = 1/opt.number_of_iterations / 10
    opt.mc_step_size = {"rS1": abs(rS1_min - rS1_max)*step_scaling,"rS2": abs(rS1_min - rS1_max)*step_scaling,"rS3": abs(rS1_min - rS1_max)*step_scaling, "kS1": abs(kS1_min - kS1_max)*step_scaling, "kS2": abs(kS2_min - kS2_max)*step_scaling, "kP": abs(kP_min - kP_max)*step_scaling, "kE1": abs(kE1_min - kE1_max)*step_scaling, "kE2": abs(kE2_min - kE2_max)*step_scaling, "alphaE": abs(alphaE_min - alphaE_max)*step_scaling, "gammaE": abs(gammaE_min - gammaE_max)*step_scaling}

    # Cold run
    start_time = time.time()
    coldrun_stoppingtemp = opt.evolve(fobj)
    
    # Record stats
    coldrun_time.append(time.time() - start_time)
    coldrun_stoptemp.append(coldrun_stoppingtemp)
    
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
    length = len([solution["rS1"] for solution in data["Solution"]])
    
    # Record stats 
    archive_cold_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[4] for v in values]
    value_2 = [v[5] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        annealed_sensfunc1.append(dummy1)
        annealed_sensfunc2.append(dummy2)
    
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
    rS1_values = [solution["rS1"] for solution in data["Solution"]]
    kS1_values = [solution["kS1"] for solution in data["Solution"]]
    kS2_values = [solution["kS2"] for solution in data["Solution"]]
    kP_values = [solution["kP"] for solution in data["Solution"]]
    kE1_values = [solution["kE1"] for solution in data["Solution"]]
    kE2_values = [solution["kE2"] for solution in data["Solution"]]
    alphaE_values = [solution["alphaE"] for solution in data["Solution"]]
    gammaE_values = [solution["gammaE"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8 in zip(rS1_values, kS1_values, kS2_values, kP_values, kE1_values, kE2_values, alphaE_values, gammaE_values):
        annealed_rS1.append(dummy1)
        annealed_kS1.append(dummy2)
        annealed_kS2.append(dummy3)
        annealed_kP.append(dummy4)
        annealed_kE1.append(dummy5)
        annealed_kE2.append(dummy6)
        annealed_alphaE.append(dummy7)
        annealed_gammaE.append(dummy8)
        
    # Create a figure with 8 1D scatter plots
    
    params = {
        "rS1": annealed_rS1,
        "kS1": annealed_kS1,
        "kS2": annealed_kS2,
        "kP": annealed_kP,
        "kE1": annealed_kE1,
        "kE2": annealed_kE2,
        "alphaE": annealed_alphaE,
        "gammaE": annealed_gammaE,
    }
    
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
    # Save rS1 annealed values
    filename = f"annealed_rS1_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_rS1)
    # Save kS1 annealed values
    filename = f"kS1_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_kS1)
    # Save kS2 annealed values
    filename = f"kS2_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_kS2)
    # Save kP annealed values
    filename = f"kP_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_kP)
    # Save kE1 annealed values
    filename = f"kE1_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_kE1)
    # Save kE2 annealed values
    filename = f"kE2_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_kE2)
    # Save alphaE annealed values
    filename = f"alphaE_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_alphaE)
    # Save gammaE annealed values
    filename = f"gammaE_values_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_gammaE)
    
    # -------------- PART 2f: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    data = pruned
        
    # Check archive length
    length = len([solution["rS1"] for solution in data["Solution"]])

    # Record stats
    archive_prune_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[4] for v in values]
    value_2 = [v[5] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_sensfunc1.append(dummy1)
        pareto_sensfunc2.append(dummy2)
    
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
    rS1_values = [solution["rS1"] for solution in data["Solution"]]
    kS1_values = [solution["kS1"] for solution in data["Solution"]]
    kS2_values = [solution["kS2"] for solution in data["Solution"]]
    kP_values = [solution["kP"] for solution in data["Solution"]]
    kE1_values = [solution["kE1"] for solution in data["Solution"]]
    kE2_values = [solution["kE2"] for solution in data["Solution"]]
    alphaE_values = [solution["alphaE"] for solution in data["Solution"]]
    gammaE_values = [solution["gammaE"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8 in zip(rS1_values, kS1_values, kS2_values, kP_values, kE1_values, kE2_values, alphaE_values, gammaE_values):
        pareto_rS1.append(dummy1)
        pareto_kS1.append(dummy2)
        pareto_kS2.append(dummy3)
        pareto_kP.append(dummy4)
        pareto_kE1.append(dummy5)
        pareto_kE2.append(dummy6)
        pareto_alphaE.append(dummy7)
        pareto_gammaE.append(dummy8)
        
    # Create a figure with 8 1D scatter plots
    
    params = {
        "rS1": annealed_rS1,
        "kS1": annealed_kS1,
        "kS2": annealed_kS2,
        "kP": annealed_kP,
        "kE1": annealed_kE1,
        "kE2": annealed_kE2,
        "alphaE": annealed_alphaE,
        "gammaE": annealed_gammaE,
    }
    
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
    # Save rS1 pareto values
    filename = f"pareto_rS1_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_rS1)
    # Save kS1 pareto values
    filename = f"kS1_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_kS1)
    # Save kS2 pareto values
    filename = f"kS2_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_kS2)
    # Save kP pareto values
    filename = f"kP_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_kP)
    # Save kE1 pareto values
    filename = f"kE1_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_kE1)
    # Save kE2 pareto values
    filename = f"kE2_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_kE2)
    # Save alphaE pareto values
    filename = f"alphaE_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_alphaE)
    # Save gammaE pareto values
    filename = f"gammaE_values_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_gammaE)
    
    # ---------------- PART 2i: SAVE MOSA DATA --------------------------------
    
    mosa_data = pd.DataFrame({
        'Circuit': [circuit],
        'Steady state': [numss],
        'Runs': [runs],
        'Random walks per run': [iterations],
        'rS1 (min, max, samples)': [[rS1_min, rS1_max, rS1_sampsize]],
        'kS1 (min, max, samples)': [[kS1_min, kS1_max, kS1_sampsize]],
        'kS2 (min, max, samples)': [[kS2_min, kS2_max, kS2_sampsize]],
        'kP (min, max, samples)': [[kP_min, kP_max, kP_sampsize]],
        'kE1 (min, max, samples)': [[kE1_min, kE1_max, kE1_sampsize]],
        'kE2 (min, max, samples)': [[kE2_min, kE2_max, kE2_sampsize]],
        'alphaE (min, max, samples)': [[alphaE_min, alphaE_max, alphaE_sampsize]],
        'gammaE (min, max, samples)': [[gammaE_min, gammaE_max, gammaE_sampsize]],
        f'(min, max) sampled value of S2ss_prime': [[S2ss_prime_samps_min, S2ss_prime_samps_max]],
        f'(min, max) sampled value of Pss_prime': [[Pss_prime_samps_min, Pss_prime_samps_max]],
        f'(min, max) sampled value of ATPss_prime': [[ATPss_prime_samps_min, ATPss_prime_samps_max]],
        f'(min, max) sampled value of Ess_prime': [[Ess_prime_samps_min, Ess_prime_samps_max]],
        f'(min, max) sampled value of {label1}': [[sens1_samps_min, sens1_samps_max]],
        f'(min, max) sampled value of {label2}': [[sens2_samps_min, sens2_samps_max]],
        f'Sampled energy difference in deltaE_S2ss': [deltaE_S2ss_prime],
        f'Sampled energy difference in deltaE_Pss': [deltaE_Pss_prime],
        f'Sampled energy difference in deltaE_ATPss': [deltaE_ATPss_prime],
        f'Sampled energy difference in deltaE_Ess': [deltaE_Ess_prime],
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
        'Step scaling factor': [step_scale],
        'Cold run time': [coldrun_time],
        'Cold run stopping temperature': [coldrun_stoptemp],
        'Prune time': [prune_time],
        'Archive length after cold run': [archive_cold_len],
        'Archive length after prune': [archive_prune_len]
    })

file_exists = os.path.exists(output_file)
mosa_data.to_csv(output_file, mode='a', index=False, header=not file_exists)

print(f"Appended new data to {output_file}")

    
    # -------------- UP TO HERE-------------------- UP TO HERE-------------------- UP TO HERE-------------------- UP TO HERE---------------------