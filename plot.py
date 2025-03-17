import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import re
import numpy as np

species = {
    1: "OneSpecies",
    2: "TwoSpecies"
}

print("""
        Choose number of species:
            1. OneSpeciesCircuits
            2. TwoSpeciesCircuits
      """)

num_species = int(input("Pick number of species: "))

os.chdir(f"{species[num_species]}Circuits")

dirlist = [item for item in os.listdir(os.getcwd()) if (os.path.isdir(os.path.join(os.getcwd(), item)) and item != "__pycache__")]

print("\n")
dirs = {}
for i, item in enumerate(dirlist):
    dirs[i+1] = item
    print(f"{i+1}. {item}")

circuit = int(input("\nPick a circuit: "))

os.chdir(dirs[circuit])

circuitlist = [item for item in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), item))]

print("\n")
circuits = {}
for i, item in enumerate(circuitlist):
    circuits[i+1] = item
    print(f"{i+1}. {item}")

sens = int(input("\nPick functions: "))

f = f"{circuits[sens]}/{species[num_species]}_{circuits[sens][15:]}.csv"

df = pd.read_csv(f)

cols = df.columns.tolist()

for i, col in enumerate(cols):
    print(f"{i+1}. {col}")

pairs = input("\nChoose pairs to plot (i.e. x1 y1, x2 y2): ").split(",")

fig, ax = plt.subplots((len(pairs)+1)//2, 2, figsize=(10,5*(len(pairs)+1)//2))

ax = ax.flatten()

array_like_pattern = re.compile(r'^\s*\[.*\]\s*$')
def conditional_literal_eval(val):
    if isinstance(val, str) and array_like_pattern.match(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    return val

def is_single_number_or_single_element(value):
    if isinstance(value, (int, float, np.number)):  # Single numbers
        return True
    elif isinstance(value, (list, np.ndarray)) and len(value) == 1:  # Single-element lists/arrays
        return True
    return False

for i, pair in enumerate(pairs):
    var1, var2 = map(int, pair.split())

    x = pd.Series(df[cols[var1-1]].apply(conditional_literal_eval))
    y = pd.Series(df[cols[var2-1]].apply(conditional_literal_eval))
    
    if x.isnull().all() or y.isnull().all():
        print(f"Skipping plot for {cols[var1-1]} vs {cols[var2-1]} due to empty data.")
        continue
    
    x = x.dropna().iloc[0]
    y = y.dropna().iloc[0]

    for index, row in df.iterrows():
        if is_single_number_or_single_element(x):           # Runs
            x = range(1, df[cols[var1-1]][0]+1)
        elif is_single_number_or_single_element(y):           # Runs
            y = range(1, df[cols[var2-1]][0]+1)
        ax[i].plot(x, y, label=f"{cols[var1-1]} vs {cols[var2-1]}", linestyle='-')
    ax[i].set_xlabel(f"{cols[var1-1]}")
    ax[i].set_ylabel(f"{cols[var2-1]}")

for j in range(i + 1, len(ax)):
    if j < len(ax): 
        fig.delaxes(ax[j])

try:
    plt.tight_layout()
    plt.show()
except Exception as e:
    plt.show()