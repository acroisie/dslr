import sys
import csv
from math_utils import MathUtils
from describe import Dataset

def load_weights(weights_path):
    thetas_by_house = {}
    selected_features = []
    try:
        with open(weights_path, "r") as weights_file:
            reader = csv.DictReader(weights_file)
            selected_features = reader.fieldnames[1:]
            for row in reader:
                house = row["House"]
                thetas_by_house[house] = [float(row[feature]) for feature in selected_features]
    except FileNotFoundError:
        print(f"File {weights_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    
    print(f"Loaded weights for houses: {list(thetas_by_house.keys())}")
    print(f"Selected features: {selected_features}")
    return thetas_by_house, selected_features

def build_X_test(dataset, selected_features):
    data = dataset.get_data()

    means, stds = MathUtils.calculate_norm_params(data, selected_features)

    normalized_data = MathUtils.normalize_data(data, selected_features, means, stds)

    X_test = []
    indexes = []
    for i, row in enumerate(normalized_data):
        feature_row = [1.0]
        valid_row = True
        for feature in selected_features:
            if feature in row and row[feature] != "":
                try:
                    feature_row.append(float(row[feature]))
                except ValueError:
                    valid_row = False
                    break
            else:
                valid_row = False
                break

        if valid_row:
            X_test.append(feature_row)
            indexes.append(i)

    print(f"Built X_test with {len(X_test)} rows")
    return X_test, indexes

def predict_houses(X_test, thetas_by_house):
    predictions = []
    houses_list = list(thetas_by_house.keys())

    for x in X_test:
        best_house = None
        best_probability = -1
        for house in houses_list:
            theta = thetas_by_house[house]
            probability = MathUtils.compute_tendency(x, theta)
            if probability > best_probability:
                best_probability = probability
                best_house = house
        predictions.append(best_house)

    print(f"Predicted houses for {len(predictions)} students")
    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_test.csv> <weights.csv>")
        sys.exit(1)

    dataset_test_path = sys.argv[1]
    weights_path = sys.argv[2]
    thetas_by_house, selected_features = load_weights(weights_path)
    test_dataset = Dataset(dataset_test_path)
    X_test, indexes = build_X_test(test_dataset, selected_features)
    predictions = predict_houses(X_test, thetas_by_house)
    with open("houses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for idx, house in zip(indexes, predictions):
            writer.writerow([idx, house])

    print("Predictions written to houses.csv")