import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')


def load_data(filename, data_path=DATA_PATH, separator=';', filetype='csv', skiprows=0):
	file_path = os.path.join(data_path, filename)
	data = None
	if filetype == 'csv':
		data = pd.read_csv(file_path, sep=separator)
	elif filetype == 'dat':
		data = pd.read_table(file_path, header=None, skiprows=skiprows, sep=separator, engine='python')
	return data


def save_data(data_frame, filename, data_path=DATA_PATH):
	csv_path = os.path.join(data_path, filename)
	return data_frame.to_csv(csv_path, float_format='%.3f', index=False)


def split_train_test(X, y, test_size=0.1, random_seed=21):
	np.random.seed(random_seed)
	shuffled_indices = np.random.permutation(len(X))
	set_size = int(len(X) * test_size)
	
	test_indices = shuffled_indices[:set_size]
	train_indices = shuffled_indices[set_size:]
	
	return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


def format_dataset(data_frame, output_attributes_names, shuffle=False):
	'''
	Arguments:
	----------
	data -- original dataset
	output_attributes_names -- list, contains the names of the output attributes
	'''
	if shuffle:
		data_frame = shuffle_dataset(data_frame)

	X = data_frame.loc[:, data_frame.columns != output_attributes_names]
	y = data_frame.loc[:, data_frame.columns == output_attributes_names]

	return X, y


def shuffle_dataset(data_frame):
	result = data_frame.iloc[np.random.permutation(len(data_frame))]
	return result


def load_iris_data(fold=1, data_type='train'):
	t = 'tra' if data_type == 'train' else 'tst'
	dataset = load_data(f'iris/10-fold/iris-10-{fold}{t}.dat', filetype='dat', skiprows=9, separator=',')

	dataset[4] = dataset[4].map({' Iris-setosa': 1.0, ' Iris-versicolor': 2.0, ' Iris-virginica': 3.0})

	dataset = dataset.rename(columns={0: 'SepalLength', 1: 'SepalWidth', 2: 'PetalLength', 3: 'PetalWidth', 4: 'Class'})

	return format_dataset(dataset, 'Class', shuffle=True)


if __name__ == "__main__":
	pass
