from describe import Dataset
from math_utils import MathUtils
import csv
import sys

def load_data(selected_features, dataset_path):
    dataset = Dataset(dataset_path, "train")
    data = dataset.get_data()
    houses = dataset.get_houses()

    means, stds = MathUtils.calculate_norm_params(data, selected_features)

    normalized_data = MathUtils.normalize_data(data, selected_features, means, stds)

    X = []
    y = []
    for row in normalized_data:
        feature_row = [1.0]
        for feature in selected_features:
            feature_row.append(row[feature])
        X.append(feature_row)
        y.append(row["Hogwarts House"])

    thetas_by_house = {}
    for house in houses:
        y_binary = [1 if label == house else 0 for label in y]
        theta = train_logistic_regression(X, y_binary)
        thetas_by_house[house] = theta

    accuracy = compute_training_accuracy(X, y, thetas_by_house)
    print(f"Training accuracy {accuracy * 100:.2f}%")

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

def compute_training_accuracy(X, actual_houses, thetas_by_house):
    correct = 0
    total = len(X)
    for i, x in enumerate(X):
        predicted = predict_for_sample(x, thetas_by_house)
        if predicted == actual_houses[i]:
            correct += 1
    return correct / total

def predict_for_sample(X, thetas_by_house):
    houses_list = list(thetas_by_house.keys())
    best_house = None
    best_probability = -1
    for house in houses_list:
        theta = thetas_by_house[house]
        probability = MathUtils.compute_tendency(X, theta)
        if probability > best_probability:
            best_probability = probability
            best_house = house
    return best_house

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