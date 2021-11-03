import numpy as np

def save_lstm_results(train_lstm_output, val_lstm_output, test_lstm_output, save_data_args, data_info_args):
	'''Saves LSTM results to a text file.'''
	if save_data_args.save_data_mode == False:
		return
	print("")
	for i in save_data_args.train:
		x = data_info_args.train_sc[i].rfind("/") + 1
		filename = save_data_args.prefix + data_info_args.train_sc[i][x:] + ".txt"
		save_data(filename, train_lstm_output[i,:,:], save_data_args)
	for i in save_data_args.val:
		x = data_info_args.train_sc[i].rfind("/") + 1
		filename = save_data_args.prefix + data_info_args.val_sc[i][x:] + ".txt"
		save_data(filename, val_lstm_output[i,:,:], save_data_args)
	for i in save_data_args.test:
		x = data_info_args.train_sc[i].rfind("/") + 1 
		filename = save_data_args.prefix + data_info_args.test_sc[i][x:] + ".txt"
		save_data(filename, test_lstm_output[i,:,:], save_data_args)

def save_data(filename, output, save_data_args):
	my_header = "Zcg        Roll        Pitch"
	if len(save_data_args.output_path)>0 and not save_data_args.output_path[-1]=="/":
		save_data_args.output_path += "/"
		print("The user specified output_path did not have a / at the end, so I added one")
	try:
		np.savetxt(save_data_args.output_path + filename, output, fmt="%.5f" ,delimiter='     ', header=my_header)
		print("Successfully saved output in ", save_data_args.output_path + filename)
	except:
		print("Error saving file (possibly the output path). Attempting to save in current directory")
		print("output path is: ", save_data_args.output_path)
		try:
			np.savetxt(filename, output, delimiter='     ', fmt="%.5f", header=my_header)
			print("Successfully saved output in current directory: ", filename)
		except:
			print("Failed to save file called ", filename)



