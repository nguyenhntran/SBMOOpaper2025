import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from paretoset import paretoset

# Replace `run` with the correct run number (e.g., run = 0 for the first run)
run = 0

# Load data
pareto_sensfunc1 = np.load(f'data/pareto_sensfunc1_run{run+1}.npy')
pareto_sensfunc2 = np.load(f'data/pareto_sensfunc2_run{run+1}.npy')
pareto_J = np.load(f'data/pareto_J_run{run+1}.npy')

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pareto_sensfunc1, pareto_sensfunc2, pareto_J)

ax.set_xlabel('Sensitivity Function 1')
ax.set_ylabel('Sensitivity Function 2')
ax.set_zlabel('J')

plt.title(f'3D Pareto Plot (Run {run+1})')
plt.show()


#----------------------------------------------------



# Stack the two arrays as columns
pareto_front = np.column_stack((pareto_sensfunc1, pareto_sensfunc2, pareto_J))

# There may be NaNs in the array. Pareto minimisation will think NaNs are minimum. We don't want this. Let's replace NaNs with infinities.
pareto_front = np.where(np.isnan(pareto_front), np.inf, pareto_front)

# Compute mask
mask = paretoset(pareto_front, sense=["min", "min", "min"])

print(f"Run: {run}")
if sum(mask) == len(pareto_sensfunc1):
    print("is good")
