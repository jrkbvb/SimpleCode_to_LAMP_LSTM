from objective_functions import objective_function
import torch
import numpy as np

def print_error_report(train_output, val_output, test_output, train_target, val_target, test_target, args):
	realization_length = train_output.shape[1]
	if realization_length != val_output.shape[1]:
		error("training output and validation output not equal length. See Print Error Report Script")
	train_output = torch.from_numpy(train_output)
	val_output = torch.from_numpy(val_output)
	test_output = torch.from_numpy(test_output)
	train_target = torch.from_numpy(train_target)
	val_target = torch.from_numpy(val_target)
	test_target = torch.from_numpy(test_target)
	
	train_MSE_loss = objective_function(train_output, train_target, [0,0,0])
	train_amp_mag_MSE_loss = objective_function(train_output, train_target, [1, 0.1, 0.9])
	train_peak_amp_MSE_loss = objective_function(train_output, train_target, [2,0,0])
	
	val_MSE_loss = objective_function(val_output, val_target, [0,0,0])
	val_amp_mag_MSE_loss = objective_function(val_output, val_target, [1, 0.1, 0.9])
	val_peak_amp_MSE_loss = objective_function(val_output, val_target, [2,0,0])

	test_MSE_loss = objective_function(test_output, test_target, [0,0,0])
	test_amp_mag_MSE_loss = objective_function(test_output, test_target, [1, 0.1, 0.9])
	test_peak_amp_MSE_loss = objective_function(test_output, test_target, [2,0,0])
	print("--------------------------------------------")
	print("Training Data Errors:")
	print("MSE:                 ", train_MSE_loss)
	print("Amplitude Magnified: ", train_amp_mag_MSE_loss)
	print("Peak MSE:            ", train_peak_amp_MSE_loss)
	print("--------------------------------------------")
	print("Validation Data Errors:")
	print("MSE:                 ", val_MSE_loss)
	print("Amplitude Magnified: ", val_amp_mag_MSE_loss)
	print("Peak MSE:            ", val_peak_amp_MSE_loss)
	print("--------------------------------------------")
	print("Test Data Errors:")
	print("MSE:                 ", test_MSE_loss)
	print("Amplitude Magnified: ", test_amp_mag_MSE_loss)
	print("Peak MSE:            ", test_peak_amp_MSE_loss)
	print("--------------------------------------------")