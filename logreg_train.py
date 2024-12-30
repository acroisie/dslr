from describe import Dataset
from math_utils import MathUtils
import csv
import sys
import matplotlib.pyplot as plt

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
        theta = train_logistic_regression(X, y_binary, house)
        thetas_by_house[house] = theta

    accuracy = compute_training_accuracy(X, y, thetas_by_house)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Plot feature importance
    plot_feature_importance(thetas_by_house, selected_features)

    return thetas_by_house

def train_logistic_regression(X, y, house_name=None):
    feature_amount = len(X[0])
    theta = [0.0] * feature_amount
    alpha = 0.1
    iterations = 1000

    costs = []  # To track the cost function values

    for _ in range(iterations):
        gradients = MathUtils.compute_gradient(X, y, theta)
        for j in range(feature_amount):
            theta[j] -= gradients[j] * alpha

        # Compute cost and track it
        cost = MathUtils.compute_cost(X, y, theta)
        costs.append(cost)

    # Plot the cost curve for the current house
    if house_name:
        plt.plot(range(iterations), costs, label=house_name)

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

def plot_feature_importance(thetas_by_house, selected_features):
    feature_importance = [0] * len(selected_features)
    for theta in thetas_by_house.values():
        for i, weight in enumerate(theta[1:], start=0):  # Skip bias (theta[0])
            feature_importance[i] += abs(weight)

    feature_importance = [imp / len(thetas_by_house) for imp in feature_importance]  # Average
    plt.figure()
    plt.bar(selected_features, feature_importance)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Average Absolute Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)
    selected_features = ["Astronomy", "Herbology", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]
    thetas = load_data(selected_features, sys.argv[1])

    plt.figure()
    plt.title("Cost Function During Training")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

    with open("weights.csv", "w", newline="") as file:
        writer = csv.writer(file)
        nb_thetas = len(next(iter(thetas.values())))
        header = ["House"] + [f"{feature}" for feature in selected_features]
        writer.writerow(header)

        for house, theta_values in thetas.items():
            row = [house] + theta_values
            writer.writerow(row)