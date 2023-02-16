import numpy as np

def save_lstm_results(train_lstm_output, val_lstm_output, test_lstm_output, save_data_args, data_info_args, args, std_factors):
	'''Saves LSTM results to a text file.'''
	if save_data_args.save_data_mode == False:
		return
	print("")
	for i in save_data_args.train:
		x = data_info_args.train_sc[i].rfind("\\") + 1
		filename = save_data_args.prefix + data_info_args.train_sc[i][x:] + ".txt"
		time_vector = np.loadtxt(data_info_args.train_sc[i] + ".mot",skiprows=2)[:,0:1]
		save_data(filename, train_lstm_output[i,:,:], save_data_args, args, time_vector)
	for i in save_data_args.val:
		x = data_info_args.val_sc[i].rfind("\\") + 1
		filename = save_data_args.prefix + data_info_args.val_sc[i][x:] + ".txt"
		time_vector = np.loadtxt(data_info_args.val_sc[i] + ".mot",skiprows=2)[:,0:1]
		save_data(filename, val_lstm_output[i,:,:], save_data_args, args, time_vector)
	for i in save_data_args.test:
		x = data_info_args.test_sc[i].rfind("\\") + 1 
		filename = save_data_args.prefix + data_info_args.test_sc[i][x:] + ".txt"
		time_vector = np.loadtxt(data_info_args.test_sc[i] + ".mot",skiprows=2)[:,0:1]
		save_data(filename, test_lstm_output[i,:,:], save_data_args, args, time_vector)

def save_data(filename, output, save_data_args, args, time_vector):
	header_array = np.array(["Time        ", "VBM        ", "Zcg        ", "Roll        ", "Pitch        "])
	my_header = ''.join(header_array[[True, args.output_vbm, args.output_3dof, args.output_3dof, args.output_3dof]])
	if len(save_data_args.output_path)>0 and not save_data_args.output_path[-1]=="\\":
		save_data_args.output_path += "\\"
		print("The user specified output_path did not have a \\ at the end, so I added one")
	try:
		np.savetxt(save_data_args.output_path + filename, np.concatenate((time_vector[:output.shape[0]], output),axis=1), fmt="%.5f" ,delimiter='     ', header=my_header)
		# print("Successfully saved output in ", save_data_args.output_path + filename)
	except:
		print("Error saving file (possibly the output path). Attempting to save in current directory")
		print("output path is: ", save_data_args.output_path)
		try:
			np.savetxt(filename, np.concatenate((time_vector[:output.shape[0]], output),axis=1), delimiter='     ', fmt="%.5f", header=my_header)
			print("Successfully saved output in current directory: ", filename)
		except:
			print("Failed to save file called ", filename)

def save_lstm_test_results(test_lstm_output, save_data_args, data_info_args, args, std_factors):
	for i in save_data_args.test:
		x = data_info_args.test_sc[i].rfind("\\") + 1 
		filename = save_data_args.prefix + data_info_args.test_sc[i][x:] + ".txt"
		time_vector = np.loadtxt(data_info_args.test_sc[i] + ".mot",skiprows=2)[:,0:1]
		save_data(filename, test_lstm_output[i,:,:], save_data_args, args, time_vector)

