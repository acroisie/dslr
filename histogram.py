from describe import Dataset
import matplotlib.pyplot as plt
import sys

def on_close(event):
    print(f"\n{event.name}")
    sys.exit(0)

if __name__ == "__main__":
    dataset = Dataset("./data/dataset_train.csv")
    features = dataset.get_features()
    houses = dataset.get_houses()

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('close_event', on_close)

try:
    for feature in features:
        house_values = dataset.values_by_house(feature)
        ax.clear()

        for house in houses:
            ax.hist(house_values[house], bins=30, alpha=0.5, label=house)

        ax.set_title(f"Distribution of {feature} by Hogwarts House")
        ax.set_xlabel(feature)
        ax.set_ylabel("Number of Students")
        ax.legend()

        plt.draw()
        input("Press [enter] to show next feature...")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    plt.ioff()
    plt.close("all")
