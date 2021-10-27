class SavedLSTM(object):
	def __init__(self, model, user_input_args, data_info, wave_mean, wave_std):
		'''This class is for saving a network model and the associated settings used
		when it was trained'''
		self.model = model #the model parameters (all the weights)
		self.user_input_args = user_input_args
		self.data_info = data_info
		self.wave_mean = wave_mean
		self.wave_std = wave_std