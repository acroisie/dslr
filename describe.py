import csv
import math
import sys


class Dataset:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.read_csv()
        self.features = self.get_numerical_features()
        self.statistics = self.get_statistics()

    def read_csv(self):
        data = []
        try:
            with open(self.filename, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)
        if not data:
            print(f"File {self.filename} is empty.")
            sys.exit(1)

        return data

    def get_numerical_features(self):
        numerical_features = []
        first_row = self.data[0]
        for feature_name in first_row:
            if feature_name == "Index":
                continue
            value = first_row[feature_name]
            if value != "":
                try:
                    float(first_row[feature_name])
                    numerical_features.append(feature_name)
                except ValueError:
                    continue

        return numerical_features

    def get_statistics(self):
        stats = {}
        for feature in self.features:
            values = []
            for row in self.data:
                value = row[feature]
                if value != "":
                    try:
                        values.append(float(value))
                    except ValueError:
                        continue

            stats[feature] = {
                "Count": len(values),
                "Mean": self.mean(values),
                "Std Dev": self.std_dev(values),
                "Min": self.find_min(values),
                "25%": self.get_percentiles(values)[25],
                "50%": self.get_percentiles(values)[50],
                "75%": self.get_percentiles(values)[75],
                "Max": self.find_max(values),
            }

        return stats

    def mean(self, values):
        total = 0.0
        for value in values:
            total += value
        return total / len(values)

    def std_dev(self, values):
        mean = self.mean(values)
        total = 0.0
        for value in values:
            total += (value - mean) ** 2

        return math.sqrt(total / len(values))

    def find_min(self, values):
        min_value = values[0]
        for value in values:
            if value < min_value:
                min_value = value

        return min_value

    def find_max(self, values):
        max_value = values[0]
        for value in values:
            if value > max_value:
                max_value = value

        return max_value

    def get_percentiles(self, values):
        if not values:
            return {25: None, 50: None, 75: None}

        sorted_values = sorted(values)
        n = len(sorted_values)
        percentiles = {}
        for percentile in [25, 50, 75]:
            k = (n - 1) * percentile / 100
            f = math.floor(k)
            c = math.ceil(k)
            if c == f:
                value = sorted_values[int(k)]
            else:
                d0 = sorted_values[int(f)] * (c - k)
                d1 = sorted_values[int(c)] * (k - f)
                value = d0 + d1
            percentiles[percentile] = value

        return percentiles

    def truncate(self, string, lenght):
        if len(string) > lenght:
            return string[: lenght - 2] + ".."
        else:
            return string

    def display_statistics(self):
        rows = ["Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max"]
        column_width = 14
        headers = [""]

        for feature in self.features:
            headers.append(self.truncate(feature, column_width))

        for header in headers:
            print(f"{header:<{column_width}}", end="|")
        print()

        for row in rows:
            print(f"{row:<{column_width}}", end="|")
            for feature in self.features:
                value = self.statistics[feature][row]
                if isinstance(value, float):
                    print(f"{value:<{column_width}.6f}", end="|")
                else:
                    print(f"{value:<{column_width}}", end="|")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset>")
        sys.exit(1)
    else:
        dataset = Dataset(sys.argv[1])
        dataset.display_statistics()
