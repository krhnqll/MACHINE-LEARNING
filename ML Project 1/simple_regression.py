import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


def read_data(file_name):
    data = np.genfromtxt(file_name)
    X = data[:, :-1] 
    y = data[:, -1]  
    return X, y


def train(X, y, lamda, epochs):
    X = np.c_[np.ones(X.shape[0]), X]

    w = np.zeros(X.shape[1])

    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        y_pred = X @ w

        error = y_pred - y

        gradient = (X.T @ error) / m

        w -= lamda * gradient

        cost = compute_cost(X, y, w)
        cost_history.append(cost)

    return w, cost_history


def compute_rmse(X, y, w):
    X = np.c_[np.ones(X.shape[0]), X]
    y_pred = X @ w
    error = y_pred - y
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse


def compute_cost(X, y, w):
    m = len(y)
    y_pred = X @ w
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


def compute_gradient(X, y, w):
    error = y_pred - y
    
    y_pred = X @ w  

    grad = (X.T @ error) / len(y)

    return grad


Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

mean, std = scaling.mean_std(Xtrain)
Xtrain = scaling.standardize(Xtrain, mean, std)
Xtest = scaling.standardize(Xtest, mean, std)

learning_rate = 0.1
epochs = 500
w, cost_history = train(Xtrain, ttrain, learning_rate, epochs)

print("Learned parameters (w):", w)

plt.plot(range(epochs), cost_history)
plt.xlabel("Number of epochs")
plt.ylabel("Cost J(w)")
plt.title("Cost J(w) vs. Number of epochs")
plt.show()

Xtrain_bias = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
w_normal_eq = np.linalg.inv(Xtrain_bias.T @ Xtrain_bias) @ (Xtrain_bias.T @ ttrain)
print("Solution from normal equations (w):", w_normal_eq)

rmse = compute_rmse(Xtest, ttest, w)
print("RMSE on test data:", rmse)

plt.scatter(Xtrain[:, 0], ttrain, color="blue", label="Training data")
plt.scatter(Xtest[:, 0], ttest, color="green", label="Test data", marker="x")

x_values = np.linspace(min(Xtrain[:, 0]), max(Xtrain[:, 0]), 100)
y_values = w[0] + w[1] * x_values
plt.plot(x_values, y_values, color="red", label="Linear solution")

plt.xlabel("Floor size")
plt.ylabel("House price")
plt.legend()
plt.title("Linear regression fit")
plt.show()