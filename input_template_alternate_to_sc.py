import numpy as np
class UserInputArgs(object):
	def __init__(self):
		self.training_mode = True #set to True to train. Otherwise skip training and just loads a previous one
		self.model_load_filename = "recently_trained_model" # "recently_trained_model" is default and
		#always used if training_mode=True. Specify something else if you want and training_mode=False.
		self.model_save_filename = "recently_trained_model" #the name of the file that will be saved as class SavedLSTM.
		#model is only saved when training_mode=True
		self.input_option = 3 #1=Just SimpleCode; 2=Just Wave Height; 3=SimpleCode and Wave Height
		self.time_res = 9 #integers only, Time resolution; 1 means every point, 2 means every other, etc.
		self.seq_length = 17990//self.time_res
		self.hidden_size = 30
		self.num_layers = 2
		self.num_batches = 9 #number of batches in each training epoch
		self.bi_directional = False
		self.dropout = 0
		self.lr = 0.005
		self.epochs = 10000 #maximum
		self.train_fun_hyp = [0, 0, 0] 
		self.val_fun_hyp   = [0, 0, 0] # First index is type of loss function.
		# Others are hyper parameters pertaining to the loss function specified.
		# 0 = MSELoss (normal)
		# 1 = amplified MSELoss
		# 2 = Peak MSELoss
		
		#For the wave grid, we want the grid to stay attached to the ship, in terms of its orientation
		#The easiest way to do this is to generate the data such that the ship's angle is 0 degrees
		#(facing positive x direction), and the beam runs parallel to the y-axis. Then you can just
		#adjust the sea heading to change the relative angle between ship and waves. These parameters
		#assume this setup. If altered, then the wave grid will need an adjustment when calculated.
		self.wave_grid_x_size = 1 #must be odd. Number of points from bow to stern
		self.wave_grid_y_size = 1 #must be odd. Number of points from stbd to port
		#if both above are =1, then the center point goes at the ship CG. 
		#if >1, then the points will be spaced such that the grid extends to the bow&stern/port&stbd.
		#in this latter case, the center point will probably not be the ship CG.

		if self.input_option==1:
			self.input_size = 3 #Zcg, Roll, Pitch
		elif self.input_option==2:
			self.input_size = self.wave_grid_x_size*self.wave_grid_y_size #Wave Height
		elif self.input_option==3:
			self.input_size = 3 + self.wave_grid_x_size*self.wave_grid_y_size #Zcg, Roll, Pitch, & Wave Height
		self.output_size = 3 #Zcg, Roll and Pitch; 3 by default
		

class PlottingArgs(object):
	def __init__(self):
		self.plotting_mode = True #set to false to not plot anything
		'''
		Each item in the lists below should be an integer that correlates to the index of the realization
		that you want plotted from the DataInfo Class (see down below). For example: putting a 0 in 
		self.prediction_ID_list_test will plot the first realization in the test set.
		'''
		self.prediction_ID_list_train = [] #for plotting the LSTM heave, roll, pitch and compare to SC & LAMP
		self.prediction_ID_list_val   = []
		self.prediction_ID_list_test  = [0]

		self.error_ID_list_train      = [] #for the error plots
		self.error_ID_list_val        = []
		self.error_ID_list_test       = [0]

		self.maxima_ID_list_train     = [0,1,2,3,4] #for the scatter plot of maxima obtained. Just 1 plot for all listed here.
		self.maxima_ID_list_val       = [0,1,2,3,4] 
		self.maxima_ID_list_test      = [0] 

		self.simple_color = "red"
		self.lamp_color = "black"
		self.lstm_color = "green"

		self.running_avg_span = 100 #to the left and to the right
		self.maxima_bullet_size = 20

class DataInfoArgs(object):
	def __init__(self):
		'''
		for this class, you can set everything manually, or set it up in some loops, depending on file names.
		This class essentially contains six lists of strings that are all the filepaths to get to the .mot and .sea files.
		The .sea files should have the same file names as the lamp .mot files, with the exception of ending in .sea instead
		of .mot. They should also be in the same folder as the lamp files.
		Don't include ".mot" and ".sea" at the end of the string
		'''
		self.train_lamp = []
		self.train_sc   = [] #sc = SimpleCode

		self.val_lamp   = []
		self.val_sc     = []

		self.test_lamp  = []
		self.test_sc    = []

		### modify this next part as necessary to fill in the lists above ###
		train_records = ["5", "6", "7", "8", "9"]
		val_records = ["10", "11", "12", "13", "14"]
		test_records = ["15", "16", "17", "18"]
		path = "C:/Users/danci/Documents/MIT/Thesis/Data_for_MIT/alternate_to_sc/"

		# train_records = ["01", "02", "03", "04", "05"]
		# val_records = ["06", "07", "08", "09"]
		# test_records = ["10"]
		# path = "C:/Users/danci/Documents/MIT/Thesis/Data_for_MIT/Set_1_forwardmoving_headon_waves/"

		simple_prefix = "data_adhoc_"
		lamp_prefix = "L1_ONRFL_IR"
		#training
		for i in train_records:
			self.train_sc.append(path+simple_prefix+i)
			self.train_lamp.append(path+lamp_prefix+i)
		#validation
		for i in val_records:
			self.val_sc.append(path+simple_prefix+i)
			self.val_lamp.append(path+lamp_prefix+i)
		#testing
		for i in test_records:
			self.test_sc.append(path+simple_prefix+i)
			self.test_lamp.append(path+lamp_prefix+i)

class SaveDataArgs(object):
	def __init__(self):
		self.save_data_mode = True
		self.output_path = "C:/Users/danci/Documents/MIT/Thesis/Data_for_MIT/alternate_to_sc/" #if the output_path="", then it will save in the currently running directory.
		# if the output_path is invalid, then it will save the file in the currently running directory.
		self.prefix = "lstm_output_for_" #prefix for filename
		#these are the indices of the training, validation, and test sets that you want the LSTM output to be saved to a text file.
		self.train = []
		self.val   = []
		self.test  = [0]
		
class DerivedArgs(object):
	def __init__(self, usr_args, data_info):
		'''The user does not need to put anything in here. These are derived from the other input arguments'''
		self.num_train_realizations = len(data_info.train_lamp)
		self.num_val_realizations   = len(data_info.val_lamp)
		self.num_test_realizations  = len(data_info.test_lamp)
		self.num_realizations = self.num_train_realizations + self.num_val_realizations + self.num_test_realizations
		self.train_batch_size = -(-(self.num_train_realizations*usr_args.time_res)//usr_args.num_batches) #ceiling division
		self.val_batch_size = self.num_val_realizations*usr_args.time_res
		self.test_batch_size = self.num_test_realizations*usr_args.time_res
