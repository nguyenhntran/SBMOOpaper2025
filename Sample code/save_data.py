import pandas as pd
import random
import os

runs = 10
csv_file = "df.csv"

# Generate new data
hot = []
cold = []
prune = []

for run in range(1, runs + 1):
    random.seed(run)
    hot.append(random.random())
    cold.append(random.random())
    prune.append(random.random())

# Create a DataFrame for the new data
print({
    'runs': [runs],
    'hot': [hot],
    'cold': [cold],
    'prune': [prune]
})

new_data = pd.DataFrame({
    'runs': [runs],
    'hot': [hot],
    'cold': [cold],
    'prune': [prune]
})

# Append new data to the CSV without loading the full file into memory
file_exists = os.path.exists(csv_file)
new_data.to_csv(csv_file, mode='a', index=False, header=not file_exists)

print(f"Appended new data to {csv_file}")