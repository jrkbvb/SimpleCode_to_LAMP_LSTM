import numpy as np
def load_fullseries(num_inputs, args, simple_filenames, lamp_filenames):
	lstm_inputs = []
	target_outputs = []
	num_files = len(simple_filenames)
	num_truncate = 10 #skipping last ten-ish rows because they go to all zero in the SIMPLE file for some reason
	for k in range(num_files):
		simple_filename = simple_filenames[k]
		lamp_filename   = lamp_filenames[k]
		if args.input_option==2 or args.input_option==3:
			if args.wave_grid_x_size==1 and args.wave_grid_y_size==1:
				wave_content = np.loadtxt(lamp_filename + ".wav", skiprows=3)
				wave_data = wave_content[:-num_truncate, 1:2]
			else:
				wave_data = np.loadtxt(lamp_filename + ".wave_grid")[:-num_truncate, :]
		simple_content = np.loadtxt(simple_filename + ".mot",skiprows=2)
		lamp_content = np.loadtxt(lamp_filename + ".mot", skiprows=3)
		simple_data = simple_content[:-num_truncate, [3,4,5]] # the 3,4,and 5 columns (4th, 5th, and 6th) are Zcg, roll, and Pitch										
		lamp_data = lamp_content[:-num_truncate, [3,4,5]]
		if args.input_option == 1:
			lstm_inputs.append(simple_data)
		else:
			#We have to keep the simple_data in for the moment for standardization purposes
			lstm_inputs.append(np.concatenate((simple_data, wave_data), axis=1))
		target_outputs.append(lamp_data)
	lstm_inputs = np.asarray(lstm_inputs)
	target_outputs = np.asarray(target_outputs)	
	return lstm_inputs, target_outputs

def load_and_standardize(simple_filenames, lamp_filenames, args, wave_mean=None, wave_std=None):
	lstm_inputs, target_outputs = load_fullseries(args.input_size, args, simple_filenames, lamp_filenames)
	num_datasets = lstm_inputs.shape[0]
	flag=False
	if wave_mean==None and wave_std==None:
		flag = True
		if args.input_option==1:
			wave_mean = 0
			wave_std  = 1
		else:
			wave_mean = np.mean(lstm_inputs[:, :, 3:])
			wave_std  = np.std(lstm_inputs[:, :, 3:])
	for i in range(num_datasets):
		zcg_mean = np.mean(lstm_inputs[i:(i+1), :, 0])
		zcg_std  = np.std(lstm_inputs[i:(i+1), :, 0])
		roll_mean = np.mean(lstm_inputs[i:(i+1), :, 1])
		roll_std  = np.std(lstm_inputs[i:(i+1), :, 1])
		if roll_std<=.00001:
			roll_std=1 #for head on waves, all roll is zero, no need to standardize
		pitch_mean = np.mean(lstm_inputs[i:(i+1), :, 2])
		pitch_std  = np.std(lstm_inputs[i:(i+1), :, 2])
		
		lstm_inputs[i:i+1,:,0] = (lstm_inputs[i:i+1,:,0]-zcg_mean)/zcg_std
		lstm_inputs[i:i+1,:,1] = (lstm_inputs[i:i+1,:,1]-roll_mean)/roll_std
		lstm_inputs[i:i+1,:,2] = (lstm_inputs[i:i+1,:,2]-pitch_mean)/pitch_std
		if not args.input_option==1:
			lstm_inputs[i:i+1,:,3:] = (lstm_inputs[i:i+1,:,3:]-wave_mean)/wave_std
		target_outputs[i:i+1,:,0]   = (target_outputs[i:i+1,:,0]-zcg_mean)/zcg_std
		target_outputs[i:i+1,:,1]   = (target_outputs[i:i+1,:,1]-roll_mean)/roll_std
		target_outputs[i:i+1,:,2]   = (target_outputs[i:i+1,:,2]-pitch_mean)/pitch_std

		sc_inputs = lstm_inputs[:,:,:3]
	if args.input_option==1:
		lstm_inputs = lstm_inputs[:,:,:3]
	elif args.input_option==2:
		lstm_inputs = lstm_inputs[:,:,3:]
	print("full series simple (input) shape ", lstm_inputs.shape)
	print("full series lamp (target) shape ", target_outputs.shape)
	if flag:
		return lstm_inputs, target_outputs, wave_mean, wave_std, sc_inputs
	else:
		return lstm_inputs, target_outputs, sc_inputs

