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

    house_colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Hufflepuff": "orange",
    }

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close)

    try:
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feature_x = features[i]
                feature_y = features[j]

                ax.clear()
                for house in houses:
                    x_values = []
                    y_values = []
                    for row in dataset.get_data():
                        if row["Hogwarts House"] == house:
                            try:
                                x_val = float(row[feature_x])
                                y_val = float(row[feature_y])
                                x_values.append(x_val)
                                y_values.append(y_val)
                            except ValueError:
                                continue

                    ax.scatter(
                        x_values,
                        y_values,
                        alpha=0.5,
                        label=house,
                    )

                ax.set_title(f"{feature_x} vs {feature_y}")
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                ax.legend()
                plt.draw()

                input("Press [enter] to show the next pair of features...")

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        plt.ioff()
        plt.close("all")
