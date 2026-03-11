import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation

# dataset 3
X = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
Y = np.array([7.46, 6.77, 12.74, 7.11, 8.81, 8.84, 6.08, 5.39, 8.15, 6.40, 5.73])

# ---------- REGRESSION LINEAIRE ----------

X_matrix = np.column_stack((np.ones(len(X)), X))
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y
b = theta[0]
a = theta[1]

print("Numpy régression linéaire")
print(f"pente a = {a}")
print(f"biais b = {b}\n")


# ---------- GRADIENT DESCENT ----------

lr = 0.01
iterations = 200
w = 0
b = min(zip(X, Y))[1]

loss_history = []
w_history = []
b_history = []

for i in range(iterations):
    y_pred = w * X + b

    loss = np.mean((Y - y_pred) ** 2)
    loss_history.append(loss)

    dw = (-2 / X.size) * np.sum(X * (Y - y_pred))
    db = (-2 / X.size) * np.sum(Y - y_pred)

    w = w - lr * dw
    b = b - lr * db

    w_history.append(w)
    b_history.append(b)

print("Gradient Descent")
print("w =", w)
print("b =", b)


# ---------- SCIKIT-LEARN ----------

model = LinearRegression()
model.fit(X.reshape(-1,1), Y)
print("\nScikit-learn")
print("coef =", model.coef_[0])
print("intercept =", model.intercept_)



plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Courbe de convergence de la descente de gradient")
plt.show()


fig, ax = plt.subplots()
ax.scatter(X, Y, color="blue")
line, = ax.plot([], [], 'r')

def update(i):
    y_pred = w_history[i]*X + b_history[i]
    line.set_data(X, y_pred)
    return line,

ani = FuncAnimation(fig, update, frames=len(w_history), interval=100)

plt.title("Trajectoire de la droite pendant GD")
plt.show()
