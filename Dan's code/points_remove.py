import numpy as np

a = np.array([[0.09+1e-10], [0.1+1e-10], [0.11+1e-12], [0.17+1e-10], [1], [1+1e-11]])  
b = np.array([[0.3+1e-12], [0.15+1e-11], [0.11+1e-11], [0.09+1e-12], [0.05+1e-11], [0.05+1e-10]])  

result = np.hstack((a, b))

dx = np.diff(result[:, 0])  # Difference in x 
dy = np.diff(result[:, 1])  # Difference in y 


threshold = 10  
ratios_xy = np.abs(dx / dy)

large_change_index = np.where(ratios_xy >= threshold)[0]

if large_change_index.size > 0:
    remove_index = large_change_index[0] + 1 
    filtered_result = result[:remove_index, :]
else:
    filtered_result = result

print("Original result:")
print(result)
print("\nFiltered result:")
print(filtered_result)