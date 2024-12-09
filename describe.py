import csv

class Dataset:
	def __init__(self, filename):
		self.filename = filename
		self.data = self.read_csv()
		self.features = self.get_numerical_features()
		self.statistics = {}


	def read_csv(self):
		data = []
		with open(self.filename, 'r') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				if row != '':
					data.append(row)
		return data
	
	def get_numerical_features(self):
		first_row = self.data[0]
		features = []
		for feature_name in first_row:
			try:
				float(first_row[feature_name])
				features.append(feature_name)
			except:
				pass