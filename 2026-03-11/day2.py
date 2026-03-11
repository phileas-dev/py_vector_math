import numpy as np

# dataset 3
X = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
Y = np.array([7.46, 6.77, 12.74, 7.11, 8.81, 8.84, 6.08, 5.39, 8.15, 6.40, 5.73])

X_matrix = np.column_stack((np.ones(len(X)), X))
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y
b = theta[0]
a = theta[1]

print(f"pente a = {a}")
print(f"biais b = {b}")

# pente a = 0.5182727272727278
# biais b = 2.9246363636363584