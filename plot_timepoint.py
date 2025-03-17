import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import re
import numpy as np
import itertools

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

print("\n")
f = f"{circuits[sens]}/{species[num_species]}_{circuits[sens][15:]}.csv"

df = pd.read_csv(f)

cols = df.columns.tolist()

fig, ax = plt.subplots(1, 2, figsize=(16,10))

array_like_pattern = re.compile(r'^\s*\[.*\]\s*$')
def conditional_literal_eval(val):
    if isinstance(val, str) and array_like_pattern.match(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    return val

def plot(yaxis, df, ax, index):
    xcol = 2
    if yaxis == "Running Time":
        ycol = [16, 20, 22]
    else:
        ycol = [24]

    x = range(1, df[cols[xcol]][0]+1)
    if len(ycol) > 1:
        y_hot = df[cols[16]].apply(conditional_literal_eval)
        y_cold = df[cols[20]].apply(conditional_literal_eval)
        y_prune = df[cols[22]].apply(conditional_literal_eval)
        y = [np.array(h) + np.array(c) + np.array(p) 
                 for h, c, p in zip(y_hot, y_cold, y_prune)][0]
        y = list(itertools.accumulate(y))
    else:
        y = np.array(df[cols[ycol[0]]].apply(conditional_literal_eval))[0]

    ax[index].plot(x, y)
    ax[index].set_xlabel("Runs")
    ax[index].set_ylabel(f"{yaxis}")
    ax[index].set_title(f"Runs vs {yaxis}")

plot("Running Time", df, ax, 0)
plot("Data Points", df, ax, 1)

plt.show()