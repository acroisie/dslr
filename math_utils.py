import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def compute_tendency(X, theta):
    z = sum(X[i] * theta[i] for i in range(len(theta)))
    return sigmoid(z)

def compute_cost(X, y, theta)