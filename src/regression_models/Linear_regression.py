import numpy as np
import matplotlib.pyplot as plt


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, convergence_tol=1e-6, n_iterations=10000):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.n_iterations = n_iterations
        self.W = None #weights
        self.b = None #intercept (bias)
        self.loss_history_ = None
        self.r2score_ = None

    def initialize_parameters(self, n_features):
        """
        Initialize model parameters.

        Parameters:
            n_features (int): The number of features in the input data.
        """
        self.W = np.zeros(n_features)
        self.b = 0

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        costs = []
        for i in range(self.n_iterations):
            #forward pass
            y_pred = np.dot(X, self.W) + self.b # wX + b = y_pred

            #the mean squared error cost.
            cost = np.sum((y - y_pred)**2) / n_samples
            #Compute gradients for model parameters (backward pass)
            dW = np.dot(X.T, y_pred - y) / n_samples
            db = np.sum(y_pred - y) / n_samples
            #Update w and b
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db


            costs.append(cost)

            if i % 100 == 0:
                print(f'Iteration: {i}, Cost: {cost}')

            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break
        self.loss_history_ = costs
        print("X shape:", X.shape)
        print("dW shape:", dW.shape)
        print("W shape:", self.W.shape)


    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def rmse(self, y, y_pred):
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def r2score(self, y, y_pred):
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))