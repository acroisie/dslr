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
        with open(self.filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data

    def get_numerical_features(self):
        numerical_features = []
        first_row = self.data[0]
        for feature_name in first_row:
            try:
                float(first_row[feature_name])
                numerical_features.append(feature_name)
            except ValueError:
                pass

        return numerical_features
    
    def get_statistics(self):
        stats = {}
        for feature in self.features:
            values = []
            for row in self.data:
                value = row[feature]
                if value != "":
                    values.append(float(value))
            stats[feature] = {
                "Count": len(values),
                "Mean": self.mean(values),
                "Std Dev": self.std_dev(values),
                "Min": self.find_min(values),
                "20%": self.get_percentiles(values)[20],
                "50%": self.get_percentiles(values)[50],
                "75%": self.get_percentiles(values)[75],
                "Max": self.find_max(values)
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
        sorted_values = sorted(values)
        n = len(sorted_values)
        percentiles = {}
        for percentile in [20, 50, 75]:
            k = (n - 1) * percentile / 100
            f = math.floor(k)
            c = math.ceil(k)
            if c == f:
                percentiles[percentile] = sorted_values[int(k)]
            else:
                d0 = sorted_values[int(f)] * (c - k)
                d1 = sorted_values[int(c)] * (k - f)
                value = d0 + d1
            percentiles[percentile] = value

        return percentiles
    
    def display_statistics(self):
        headers = [""] + list(self.features)
        rows = ["Count", "Mean", "Std Dev", "Min", "20%", "50%", "75%", "Max"]
        column_width = 15

        print(f"{'':<{column_width}}", end="")
        for feature in self.features:
            print(f"{feature:<{column_width}}", end="")
        print()

        for row in rows:
            print(f"{row:<{column_width}}", end="")
            for feature in self.features:
                value = self.statistics[feature][row]
                if isinstance(value, float):
                    print(f"{value:<{column_width}.6f}", end="")
                else:
                    print(f"{value:<{column_width}}", end="")
            print()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python describe.py <dataset>")
        sys.exit(1)
    else:
        dataset = Dataset(sys.argv[1])
        dataset.display_statistics()