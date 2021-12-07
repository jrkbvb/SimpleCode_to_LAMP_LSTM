import numpy as np

def un_standardize(target, lstm_output, sc_filepath, std_factors):
	realization_length = lstm_output.shape[0]
	simple_data = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]

	roll_mean = np.mean(simple_data[:,1])
	roll_std = np.std(simple_data[:,1])
	pitch_mean = np.mean(simple_data[:,2])
	pitch_std = np.std(simple_data[:,2])

	lamp_data = np.copy(target)
	lstm_data = np.copy(lstm_output)
	lamp_data[:,0] = target[:,0]*std_factors[3] + std_factors[2] #zcg
	lamp_data[:,1] = target[:,1]*roll_std + roll_mean
	lamp_data[:,2] = target[:,2]*pitch_std + pitch_mean
	lstm_data[:,0] = lstm_output[:,0]*std_factors[3] + std_factors[2] #zcg
	lstm_data[:,1] = lstm_output[:,1]*roll_std + roll_mean
	lstm_data[:,2] = lstm_output[:,2]*pitch_std + pitch_mean

	return simple_data, lamp_data, lstm_data

def un_standardize_output(lstm_output, sc_filepath, std_factors):
	'''This version is used in "save_lstm_results.py"'''
	realization_length = lstm_output.shape[0]
	simple_data = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]

	roll_mean = np.mean(simple_data[:,1])
	roll_std = np.std(simple_data[:,1])
	pitch_mean = np.mean(simple_data[:,2])
	pitch_std = np.std(simple_data[:,2])

	lstm_data = np.copy(lstm_output)
	lstm_data[:,0] = lstm_output[:,0]*std_factors[3] + std_factors[2] #zcg
	lstm_data[:,1] = lstm_output[:,1]*roll_std + roll_mean
	lstm_data[:,2] = lstm_output[:,2]*pitch_std + pitch_mean

	return lstm_data