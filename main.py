import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from plot_lstm_results import plot_lstm_results
from network_models import LSTM
from train_test import train, test
from S2LDataset import S2LDataset
from print_error_report import print_error_report
#On the line below, specifiy after "from" which file the user inputs are coming from.
from input_template import UserInputArgs, PlottingArgs, DataInfoArgs, SaveDataArgs, DerivedArgs
from load_and_standardize import load_and_standardize
from reshape_for_time_resolution import reshape_for_time_resolution, reshape_full_series
from save_lstm import save_lstm_info, load_lstm_info
from save_lstm_results import save_lstm_results

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
	torch.cuda.empty_cache() #to avoid some rare errors.

# Check if using CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
args = UserInputArgs()
plot_args = PlottingArgs()
data_info_args = DataInfoArgs()
save_data_args = SaveDataArgs()
derived_args = DerivedArgs(args, data_info_args)

#either load in a saved network or create a new one for training
#creating a new one for training_mode==True can be done later, and is more convenient for optimizing hyper-parameters
if args.training_mode==False:
	network, wave_mean, wave_std = load_lstm_info(args)
	network.to(device)

# Read in and Standardize the data. Each input & target is formatted:
# [num_realizations, full series length (17990), num_parameters]
if args.training_mode==False:
	print("\nLoading Training Data")
	train_input, train_target	= load_and_standardize(data_info_args.train_sc, data_info_args.train_lamp, args, wave_mean, wave_std)
elif args.training_mode==True:
	print("\nLoading Training Data")
	train_input, train_target, wave_mean, wave_std 	= load_and_standardize(data_info_args.train_sc, data_info_args.train_lamp, args)
print("\nLoading Validation Data")
val_input,   val_target 						= load_and_standardize(data_info_args.val_sc, data_info_args.val_lamp, args, wave_mean, wave_std)
print("\nLoading Testing Data")
test_input,  test_target 						= load_and_standardize(data_info_args.test_sc, data_info_args.test_lamp, args, wave_mean, wave_std)

#START TIME_RES LOOP, HIDDEN_SIZE, AND NUM_LAYERS HERE

if args.training_mode==True:
	# create an instance of our LSTM network
	network = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.bi_directional, args.dropout).to(device)

# Reshape the data to take into account the time resolution
train_input, train_target 	= reshape_for_time_resolution(train_input, train_target, args)
val_input,   val_target 	= reshape_for_time_resolution(val_input,   val_target,   args)
test_input,  test_target 	= reshape_for_time_resolution(test_input,  test_target,  args)

# Create Dataset objects for each of our train/val/test sets
train_dataset = S2LDataset(train_input, train_target)
val_dataset   = S2LDataset(val_input, val_target)
test_dataset  = S2LDataset(test_input, test_target)

# Create a PyTorch dataloader for each train/val set. Test set isn't needed until later
train_loader = DataLoader(train_dataset, batch_size=derived_args.train_batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=derived_args.val_batch_size)
test_loader   = DataLoader(test_dataset,   batch_size=derived_args.test_batch_size)

# Display dataset information
print("\nReshaped the following data:")
print(f"train_input has shape	{train_input.shape}")
print(f"train_target has shape	{train_target.shape}")
print(f"val_input has shape 	{val_input.shape}")
print(f"val_target has shape 	{val_target.shape}")
print(f"test_input has shape 	{test_input.shape}")
print(f"test_target has shape 	{test_target.shape}")

# initialize our optimizer. We'll use Adam
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

# Begin Training
if args.training_mode==True:
	best_val_loss = float('inf')
	best_loss_counter = 0
	print("\n Beginning Training")
	for epoch in range(1, args.epochs+1):
	    train_loss = train(network, device, train_loader, optimizer, args.train_fun_hyp)
	    val_loss   = test(network, device, val_loader, args.val_fun_hyp)
	    if val_loss < 0.99*best_val_loss:
	    	best_val_loss = val_loss
	    	best_loss_counter = 0
	    	torch.save(network.state_dict(), "recently_trained_model.pt")
	    else:
	    	best_loss_counter += 1
	    if best_loss_counter > 30:
	    	break
	    print('Train Epoch: {:02d} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))
	print("Training Done\n")
	network.load_state_dict(torch.load("recently_trained_model.pt")) # restoring the best found network based on validation data
	save_lstm_info(network.state_dict(), args, data_info_args, wave_mean, wave_std)	

#Produce final LSTM output
train_loader = DataLoader(train_dataset, batch_size=derived_args.train_batch_size) #now with shuffle off so everything is ordered correctly
start_time = time.time() #to show how long it takes to run the series through
train_lstm_output	= test(network, device, train_loader, args.val_fun_hyp, derived_args.num_train_realizations, args.time_res, True)
val_lstm_output 	= test(network, device, val_loader,   args.val_fun_hyp, derived_args.num_val_realizations,   args.time_res, True)
test_lstm_output 	= test(network, device, test_loader,  args.val_fun_hyp, derived_args.num_test_realizations,  args.time_res, True)
end_time   = time.time()
print("\ntrain output shape", train_lstm_output.shape)
print("val output shape  ", val_lstm_output.shape)
print("test output shape ", test_lstm_output.shape)
print("Time to produce output for ", derived_args.num_realizations," realizations:", (end_time-start_time))

#Reshape our input and targets to be same shape as output
train_input, train_target, train_lstm_output 	= reshape_full_series(train_input, train_target, train_lstm_output, args)
val_input,   val_target, val_lstm_output 	= reshape_full_series(val_input,   val_target,   val_lstm_output, args)
test_input,  test_target, test_lstm_output 	= reshape_full_series(test_input,  test_target,  test_lstm_output, args)

#Print Final Errors
print("\nSimpleCode Error Results:")
print_error_report(train_input[:,:,:3], val_input[:,:,:3], test_input[:,:,:3], train_target, val_target, test_target, args)
print("\nLSTM Error Results:")
print_error_report(train_lstm_output, val_lstm_output, test_lstm_output, train_target, val_target, test_target, args)

#Plot Results
plot_lstm_results(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, plot_args, data_info_args)

#Save Results
save_lstm_results(train_lstm_output, val_lstm_output, test_lstm_output, save_data_args, data_info_args)

plt.show()