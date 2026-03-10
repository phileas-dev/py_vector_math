import sys
sys.dont_write_bytecode = True
from day1 import datasets, linear_regression, avg_quadratic_error
import matplotlib.pyplot as plt

n = 0   # 0-3 to select dataset
v1 = datasets[n][0]
v2 = datasets[n][1]
a, b = linear_regression(v1, v2)
mse = avg_quadratic_error(v1,v2, a, b)

plt.scatter(v1, v2, color='blue', label='Data points')
x_line = [min(v1), max(v1)]
plt.plot(x_line, [a*xi + b for xi in x_line], color='red', label=f'Regression: y={a:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Linear Regression (Dataset {n+1}), MSE = {mse:2f}')
plt.legend()
plt.grid(True)
plt.show()