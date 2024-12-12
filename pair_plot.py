from describe import Dataset
import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

def on_close(event):
    print(f"\n{event.name}")
    exit(0)

if __name__ == "__main__":
    dataset = Dataset("./data/dataset_train.csv")
    features = dataset.get_features()
    houses = dataset.get_houses()
    data = dataset.get_data()

    house_colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Hufflepuff": "orange",
    }

    n = len(features)

    fig, axes = plt.subplots(n, n, figsize=(100, 100))
    fig.canvas.mpl_connect("close_event", on_close)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 150


    value_by_houses = {}
    for house in houses:
        value_by_houses[house] = {}
        for feature in features:
            value_by_houses[house][feature] = []

    for row in data:
        house = row["Hogwarts House"]
        if house not in houses:
            continue
        for feature in features:
            val_str = row[feature]
            if val_str == "":
                continue
            try:
                val = float(val_str)
                value_by_houses[house][feature].append(val)
            except ValueError:
                continue

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.clear()

            if i == j:
                feature = features[i]
                for house in houses:
                    values = value_by_houses[house][feature]
                    ax.hist(values, bins=30, alpha=0.5, label=house, color=house_colors[house])
                if j == 0:
                    ax.set_ylabel(features[i])
                if i == n - 1:
                    ax.set_xlabel(features[j])
                if i == 0 and j == 0:
                    ax.legend()
            else:
                feature_x = features[j]
                feature_y = features[i]

                for house in houses:
                    x_vals = value_by_houses[house][feature_x]
                    y_vals = value_by_houses[house][feature_y]
                    length = min(len(x_vals), len(y_vals))
                    x_vals = x_vals[:length]
                    y_vals = y_vals[:length]

                    ax.scatter(x_vals, y_vals, alpha=0.5, label=house, color=house_colors[house])

                if j == 0:
                    ax.set_ylabel(features[i])
                if i == n - 1:
                    ax.set_xlabel(features[j])

    plt.tight_layout()
    plt.show()
