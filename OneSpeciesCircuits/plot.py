import pandas as pd
import matplotlib.pyplot as plt
import ast

csv_file = "MOSA_arneg/MOSA_sensfuncs_0_and_1/OneSpecies.csv"

df = pd.read_csv(csv_file)

df['hot'] = df['hot'].apply(ast.literal_eval)
df['cold'] = df['cold'].apply(ast.literal_eval)
df['prune'] = df['prune'].apply(ast.literal_eval)

plt.figure(figsize=(12, 7))

for index, row in df.iterrows():
    runs = range(1, len(row['hot']) + 1)
    plt.plot(runs, row['hot'], marker='o', label=f'Hot (Row {index + 1})')
    plt.plot(runs, row['cold'], marker='s', label=f'Cold (Row {index + 1})')
    plt.plot(runs, row['prune'], marker='^', label=f'Prune (Row {index + 1})')

plt.xlabel('Runs')
plt.ylabel('Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()