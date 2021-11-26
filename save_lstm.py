import pickle
from network_models import LSTM

class SavedLSTM(object):
	def __init__(self, model, user_input_args, data_info_args, std_factors):
		'''This class is for saving a network model and the associated settings used
		when it was trained'''
		self.model = model #the model parameters as a state dictionary (all the weights)
		self.user_input_args = user_input_args
		self.data_info_args = data_info_args
		self.std_factors = std_factors #wave_mean, wave_std, zcg_mean, zcg_std
		# self.wave_mean = wave_mean
		# self.wave_std = wave_std

def save_lstm_info(model, user_input_args, data_info_args, std_factors):
	saved_lstm = SavedLSTM(model, user_input_args, data_info_args, std_factors)
	# Save the file
	print("Saving this newly trained model and info in: ", user_input_args.model_save_filename,".pickle")
	pickle.dump(saved_lstm, file = open(user_input_args.model_save_filename+".pickle", "wb"))

def load_lstm_info(user_input_args):
	print("Loading LSTM Saved in: ", user_input_args.model_load_filename,".pickle")
	loaded_lstm = pickle.load(open(user_input_args.model_load_filename+".pickle", "rb"))
	args = loaded_lstm.user_input_args #these are the old args associated with the saved lstm
	network = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bi_directional, args.dropout)
	network.load_state_dict(loaded_lstm.model)
	return network, loaded_lstm.std_factors