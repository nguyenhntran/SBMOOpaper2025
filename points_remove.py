import numpy as np

a = np.array([[0.09+1e-10], [0.1+1e-10], [0.11+1e-12], [0.17+1e-10], [1], [1+1e-11]])  
b = np.array([[0.3+1e-12], [0.15+1e-11], [0.11+1e-11], [0.09+1e-12], [0.05+1e-11], [0.05+1e-10]])  

result = np.hstack((a, b))

dx = np.diff(result[:, 0])  # Difference in x 
dy = np.diff(result[:, 1])  # Difference in y 


threshold = 10  
ratios_xy = np.abs(dx / dy)
# ratios_yx = np.abs(dy / dx)

# remove_indices = np.where((ratios_xy > threshold) | (ratios_yx > threshold))[0] + 1
remove_indices_large_change = np.where(ratios_xy >= threshold)[0]+1

# Set a tolerance for points that are too close
tolerance = 1e-10

# Find points that are too close (both in x and y)
remove_indices_too_close = []

for i in range(1, len(result)):
    if np.abs(result[i, 0] - result[i-1, 0]) < tolerance and np.abs(result[i, 1] - result[i-1, 1]) < tolerance:
        remove_indices_too_close.append(i)

# Combine both remove indices
remove_indices = np.unique(np.concatenate([remove_indices_large_change, remove_indices_too_close]))

# Remove the indices
filtered_result = np.delete(result, remove_indices, axis=0)

# Output the results
print("Original result:")
print(result)
print("\nFiltered result:")
print(filtered_result)