from describe import Dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = Dataset("./data/dataset_train.csv")
    features = dataset.get_features()
    houses = dataset.get_houses()

    for feature in features:
        house_values = dataset.values_by_house(feature)

    plt.figure(figsize=(8, 6))
    for house in houses:
        plt.hist(house_values[house], bins=30, alpha=0.5, label=house)

    plt.title(f"Distribution of {feature} by Hogwarts House")
    plt.xlabel(feature)
    plt.ylabel("Number of Students")
    plt.legend()
    plt.tight_layout()
    plt.show()