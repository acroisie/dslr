from describe import Dataset
import matplotlib.pyplot as plt


def on_close(event):
    print(f"\n{event.name}")
    exit(0)


if __name__ == "__main__":
    dataset = Dataset("./data/dataset_train.csv")
    features = dataset.get_features()

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close)

    try:
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feature_x = features[i]
                feature_y = features[j]

                x_values = []
                y_values = []
                for row in dataset.get_data():
                    try:
                        x_val = float(row[feature_x])
                        y_val = float(row[feature_y])
                        x_values.append(x_val)
                        y_values.append(y_val)
                    except ValueError:
                        continue

                ax.clear()
                ax.scatter(x_values, y_values, alpha=0.5)
                ax.set_title(f"{feature_x} vs {feature_y}")
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                plt.draw()

                input("Press [enter] to show the next pair of features...")

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        plt.ioff()
        plt.close("all")
