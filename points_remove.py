import numpy as np

a = np.array([[0.09+1e-10], [0.1+1e-10], [0.11+1e-12], [1], [0.17+1e-10]])  
b = np.array([[0.3+1e-12], [0.15+1e-11], [0.11+1e-11], [0.1+1e-11], [0.01+1e-12]])  

result = np.hstack((a, b))

dx = np.diff(result[:, 0])  # Difference in x 
dy = np.diff(result[:, 1])  # Difference in y 


threshold = 10  
ratios_xy = np.abs(dx / dy)
# ratios_yx = np.abs(dy / dx)

# remove_indices = np.where((ratios_xy > threshold) | (ratios_yx > threshold))[0] + 1
remove_indices = np.where(ratios_xy > threshold)[0] + 1

filtered_result = np.delete(result, remove_indices, axis=0)

print("Original result:")
print(result)
print("\nFiltered result:")
print(filtered_result)