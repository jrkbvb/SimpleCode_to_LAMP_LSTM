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

def reshape_for_time_resolution(lstm_inputs, target_outputs, args):
	# Returns data (inputs) and labels (targets)
	data = []
	labels = []
	num_truncate = lstm_inputs.shape[1]%args.time_res
	end_idx = lstm_inputs.shape[1] - num_truncate
	data = lstm_inputs[:, :end_idx, :].reshape(-1, end_idx//args.time_res, args.input_size, order='F')
	labels = target_outputs[:, :end_idx, :].reshape(-1, end_idx//args.time_res, args.output_size, order='F')
	return data, labels

def reshape_full_series(lstm_inputs, target_outputs, lstm_outputs, args):
	num_realizations = lstm_inputs.shape[0]//args.time_res
	data = lstm_inputs.reshape(num_realizations,-1, args.input_size, order='F')
	labels = target_outputs.reshape(num_realizations,-1, args.output_size, order='F')
	outputs = lstm_outputs.reshape(num_realizations, -1, args.output_size, order='F')
	return data, labels, outputs