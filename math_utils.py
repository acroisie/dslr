import math

class MathUtils:
    @staticmethod
    def sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            e_z = math.exp(z)
            return e_z / (1.0 + e_z)

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
    def calculate_norm_params(data, features):
        means = {}
        stds = {}
        for feature in features:
            values = [float(row[feature]) for row in data if row[feature] != ""]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance)
            means[feature] = mean
            stds[feature] = std
        return means, stds

    @staticmethod
    def normalize_data(data, features, means, stds):
        normalized_data = []
        for row in data:
            normalized_row = row.copy()
            for feature in features:
                if row[feature] == "":
                    continue
                value = float(row[feature])
                if stds[feature] != 0:
                    normalized_row[feature] = (value - means[feature]) / stds[feature]
                else:
                    normalized_row[feature] = 0.0
            normalized_data.append(normalized_row)
        return normalized_data