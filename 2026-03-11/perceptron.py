import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_and = np.array([0,0,0,1])
y_or  = np.array([0,1,1,1])
lr = 0.1
iterations = 20
w = np.random.randn(2)
b = np.random.randn()


# implémentation from scratch
history = []
y = y_and
for iter in range(iterations):
    for i in range(len(X)):

        x1, x2 = X[i]
        y_true = y[i]

        weighted_sum = np.dot(w, X[i]) + b
        y_pred = 1 if weighted_sum >= 0 else 0

        error = y_true - y_pred
        w[0] += lr * error * x1
        w[1] += lr * error * x2
        b += lr * error

    history.append((w.copy(), b))

print("Poids finaux:", w)
print("Bias:", b)


# implémentation sklearn
from sklearn.linear_model import Perceptron

model = Perceptron(max_iter=1000, eta0=0.1)
model.fit(X, y_and)
print(model.coef_, model.intercept_)


# animation
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    w_i, b_i = history[i]
    x_vals = np.linspace(-0.5,1.5,100)
    y_vals = -(w_i[0]/w_i[1]) * x_vals - b_i/w_i[1]

    ax.plot(x_vals, y_vals)

    for j in range(len(X)):
        if y[j] == 1:
            ax.scatter(X[j][0], X[j][1])
        else:
            ax.scatter(X[j][0], X[j][1])

    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    ax.set_title(f"Iteration {i}")

ani = FuncAnimation(fig, animate, frames=len(history), interval=400)
ani.save("perceptron_training.gif", writer="pillow", fps=2)
plt.show()

