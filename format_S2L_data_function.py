# Dayne Howard
# 30 June 2021
# Set up the input and target data (training, validation, and test)
# to be fed into "simple_to_lamp_###.py"
# data should be saved as "train_data.py", "train_labels.py",
# "val_data.py", "val_labels.py", "test_data.py", "test_labels.py"

# For this data set up, we do the simplest scheme, which is to use
# realizations 1-6 as training data, 7-8 as validation, and 9-10 as test
# The input data is sequences of SIMPLE code data, and the target labels
# are corresponding sequences of LAMP.

import numpy as np

def get_data_and_labels(file_indices, num_inputs, time_res):
	# Returns data (inputs) and labels (targets)
	data = []
	labels = []
	simple_fullseries_temp = np.load("simple_fullseries_normalized_data.npy")
	lamp_fullseries_temp = np.load("lamp_fullseries_normalized_data.npy")
	num_truncate = (simple_fullseries_temp.shape[1])%time_res
	end_idx = simple_fullseries_temp.shape[1] - num_truncate
	data = simple_fullseries_temp[file_indices, :end_idx, :].reshape(-1, end_idx//time_res, 4, order='F')
	labels = lamp_fullseries_temp[file_indices, :end_idx, :].reshape(-1, end_idx//time_res, 3, order='F')
	return data, labels