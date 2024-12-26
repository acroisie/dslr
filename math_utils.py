import math

class MathUtils:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    @staticmethod
    def compute_tendency(X, theta):
        z = sum(X[i] * theta[i] for i in range(len(theta)))
        return MathUtils.sigmoid(z)

    @staticmethod
    def compute_cost(X, y, theta):
        m = len(X)
        total = 0.0
        for i in range(m):
            h = MathUtils.compute_tendency(X[i], theta)
            total += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
        return total / m

    @staticmethod
    def compute_gradient(X, y, theta):
        m = len(X)
        grad = [0.0] * len(theta)
        for i in range(m):
            h = MathUtils.compute_tendency(X[i], theta)
            error = h - y[i]
            for j in range(len(theta)):
                grad[j] += error * X[i][j]
        for j in range(len(theta)):
            grad[j] /= m
        return grad
    
    @staticmethod
    def normalize_data(data, features):
        means = {}
        stds = {}

        for feature in features:
            values = [row[feature] for row in data]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance)
            means[feature] = mean
            stds[feature] = std

        normalized_data = []

        for row in data:
            new_row = row.copy()
            for feature in features:
                val_str = new_row[feature]
                if val_str == "":
                    continue
                val = float(val_str)
                if stds[feature] != 0:
                    new_row[feature] = (val - means[feature]) / stds[feature]
                else:
                    new_row[feature] = 0.0
            normalized_data.append(new_row)

        return normalized_data