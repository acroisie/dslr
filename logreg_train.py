from describe import Dataset
from math_utils import MathUtils
import csv
import sys

def load_data(selected_features, dataset_path):
    dataset = Dataset(dataset_path)
    data = dataset.get_data()
    houses = dataset.get_houses()

    means, stds = MathUtils.calculate_norm_params(data, selected_features)

    normalized_data = MathUtils.normalize_data(data, selected_features, means, stds)

    X = []
    for row in normalized_data:
        feature_row = [1.0]
        for feature in selected_features:
            feature_row.append(row[feature])
        X.append(feature_row)

    thetas_by_house = {}
    for house in houses:
        y = []
        for row in normalized_data:
            if row["Hogwarts House"] == house:
                y.append(1)
            else:
                y.append(0)
        theta = train_logistic_regression(X, y)
        thetas_by_house[house] = theta

    return thetas_by_house

def train_logistic_regression(X, y):
    feature_amount = len(X[0])
    theta = [0.0] * feature_amount
    alpha = 0.1
    iterations = 1000

    for _ in range(iterations):
        gradients = MathUtils.compute_gradient(X, y, theta)
        for j in range(feature_amount):
            theta[j] -= gradients[j] * alpha

    return theta

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)
    selected_features = ["Astronomy", "Herbology", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]
    thetas = load_data(selected_features, sys.argv[1])

    with open("weights.csv", "w", newline="") as file:
        writer = csv.writer(file)
        nb_thetas = len(next(iter(thetas.values())))
        header = ["House"] + [f"{feature}" for feature in selected_features]
        writer.writerow(header)

        for house, theta_values in thetas.items():
            row = [house] + theta_values
            writer.writerow(row)