import numpy as np
import matplotlib.pyplot as plt
def load_fullseries(num_inputs, args, simple_filenames, lamp_filenames):
	lstm_inputs = []
	target_outputs = []
	num_files = len(simple_filenames)
	num_truncate = 10 #skipping last ten-ish rows because they go to all zero in the SIMPLE file for some reason
	for k in range(num_files):
		simple_filename = simple_filenames[k]
		lamp_filename   = lamp_filenames[k]
		if args.wave_grid_x_size==1 and args.wave_grid_y_size==1:
			wave_content = np.loadtxt(lamp_filename + ".wav", skiprows=3)
			wave_data = wave_content[:-num_truncate, 1:2]
		else:
			wave_data = np.loadtxt(lamp_filename + ".wave_grid")[:-num_truncate, :]
		simple_content = np.loadtxt(simple_filename + ".txt",skiprows=1)
		lamp_content = np.loadtxt(lamp_filename + ".mot", skiprows=3)
		simple_data = simple_content[:-num_truncate, [1,2]] # the 1 and 2 columns (2nd and 3rd) are Zcg and Pitch	
		column_of_zeros = np.zeros((simple_data.shape[0], 1))									
		lamp_data = lamp_content[:-num_truncate, [3,4,5]]
		lstm_inputs.append(np.concatenate((simple_data[:,0:1], column_of_zeros, simple_data[:,1:2], wave_data), axis=1))
		target_outputs.append(lamp_data)
	lstm_inputs = np.asarray(lstm_inputs)
	target_outputs = np.asarray(target_outputs)	
	return lstm_inputs, target_outputs

def load_and_standardize(simple_filenames, lamp_filenames, args, std_factors=None):
	lstm_inputs, target_outputs = load_fullseries(args.input_size, args, simple_filenames, lamp_filenames)
	num_datasets = lstm_inputs.shape[0]
	flag=False
	lstm_inputs[:,:,0] += 2
	if std_factors==None:
		flag = True
		zcg_glob_mean = np.mean(lstm_inputs[:, :, 0])
		zcg_glob_std = np.std(lstm_inputs[:, :, 0])
		roll_glob_mean = np.mean(lstm_inputs[:, :, 1])
		roll_glob_std = np.std(lstm_inputs[:, :, 1])
		if roll_glob_std<=.00001:
			roll_glob_std = 1
		pitch_glob_mean = np.mean(lstm_inputs[:, :, 2])
		pitch_glob_std = np.std(lstm_inputs[:, :, 2])
		wave_glob_mean = np.mean(lstm_inputs[:, :, 3:])
		wave_glob_std  = np.std(lstm_inputs[:, :, 3:])
		std_factors = [zcg_glob_mean, zcg_glob_std, roll_glob_mean, roll_glob_std, pitch_glob_mean, pitch_glob_std, wave_glob_mean, wave_glob_std]
	for i in range(num_datasets):		
		lstm_inputs[i:i+1,:,0] = (lstm_inputs[i:i+1,:,0]-std_factors[0])/std_factors[1]
		target_outputs[i:i+1,:,0]   = (target_outputs[i:i+1,:,0]-std_factors[0])/std_factors[1]
		lstm_inputs[i:i+1,:,1] = (lstm_inputs[i:i+1,:,1]-std_factors[2])/std_factors[3]
		target_outputs[i:i+1,:,1]   = (target_outputs[i:i+1,:,1]-std_factors[2])/std_factors[3]
		lstm_inputs[i:i+1,:,2] = (lstm_inputs[i:i+1,:,2]-std_factors[4])/std_factors[5]
		target_outputs[i:i+1,:,2]   = (target_outputs[i:i+1,:,2]-std_factors[4])/std_factors[5]
		lstm_inputs[i:i+1,:,3:] = (lstm_inputs[i:i+1,:,3:]-std_factors[6])/std_factors[7]

	sc_inputs = lstm_inputs[:,:,:3]

	if args.input_option==1:
		lstm_inputs = lstm_inputs[:,:,:3]
	elif args.input_option==2:
		lstm_inputs = lstm_inputs[:,:,3:]
	print("full series simple (input) shape ", lstm_inputs.shape)
	print("full series lamp (target) shape ", target_outputs.shape)
	if flag:
		return lstm_inputs, target_outputs, std_factors, sc_inputs
	else:
		return lstm_inputs, target_outputs, sc_inputs

