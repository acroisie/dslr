from describe import Dataset
from math_utils import MathUtils
import csv
import sys

def load_data(selected_features):
    dataset = Dataset("./data/dataset_train.csv")
    data = dataset.get_data()
    houses = dataset.get_houses()

    X = []
    for row in data:
        feature_row = [1.0]
        # print(row.keys())
        for feature_index in selected_features:
            # print(row[feature_index])
            feature_row.append(row[feature_index])
        X.append(feature_row)

    thetas_by_house = {}
    for house in houses:
        y = []
        for row in data:
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
    selected_features = ["Arithmancy", "Astronomy", "Herbology"]
    thetas = load_data(selected_features)

    with open("weights.csv", "w", newline="") as file:
        writer = csv.writer(file)
        nb_thetas = len(next(iter(thetas.values())))
        header = ["House"] + [f"Theta_{i}" for i in range(nb_thetas)]
        writer.writerow(header)

        for house, theta_values in thetas.items():
            row = [house] + theta_values
            writer.writerow(row)
