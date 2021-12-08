#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code takes the existing network and produces output for many Simplecode and wave input files.
Note that only the input type in the input_general.py code mst be changed i.e., self.input_option in 
UserInputArgs.

If you want to use this code as is, my folder structure is set up as follows:
    
    LSTM Model Location: main_folder/V_XX/beta_XXX_partition/LSTM_model_name.pickle
    (the tool padn makes it so that there is a leading zero for single digit velocities and headings have
     leading zeros out to the hundred place e.g., 0 degree heading - > beta_000// 75 degree heading ->
     beta_075 // 105 degree heading -> beta_105 // 5 kts velocity -> V_05 // 10 kt velocity -> V_10)
    
    If not, be sure to comment out the the padtools line and the vfunc np.vectorize(padn) lines

"""
import torch
import numpy as np
from input_template import UserInputArgs, DerivedArgs, DataInfoArgs
from save_lstm import load_lstm_info
from torch.utils.data import Dataset
from S2LDataset import S2LDataset
from torch.utils.data import DataLoader
from padtools import padn #comment out if padn is not in working folder
import matplotlib.pyplot as plt
import os


def main():
    vfunc = np.vectorize(padn) #comment out if padn is not in working folder
    args = UserInputArgs()
    args.training_mode = False
    
    data_info_args = DataInfoArgs()
    derived_args = DerivedArgs(args, data_info_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 30.
    beta = np.array((0.,15.,30.,45.,60.,75.,90.,105.,120.,135.,150.,165.,180.))

    num_files = np.arange(100)+31 #starting at realization 000031
    path = '/Volumes/SAMDISK/sam_transfer/V_%s/simple/' % vfunc(V,2) #This is the path in which the SimpleCode files are located
    save_path = '/Volumes/SAMDISK/lstm_runs/V_%s_specific_wave/' % vfunc(V,2) #where the LSTM produced .mot files will be placed
    simple_prefix = 'flh_irreg2-0000'
    save_prefix = 'lstm_flh-0000'
    
    for b in beta:
        simple_files=[]
        save_files = []
        for n in num_files:
            stri = path + ('beta_%s/' % vfunc(b,3)) +simple_prefix+('%s' % vfunc(n,3))
            stro = save_path + ('beta_%s/' % vfunc(b,3)) +save_prefix+('%s.mot' % vfunc(n,3))
            checker = os.path.exists(save_path + ('beta_%s/' % vfunc(b,3)) )
            if not checker:
                os.mkdir((save_path + ('beta_%s/' % vfunc(b,3))))
            simple_files.append(stri)
            save_files.append(stro)
        args.model_load_filename = ("/Volumes/SAMDISK/sam_transfer/V_%s/beta_%s_partition/recently_trained_model_wave" % (vfunc(V,2),vfunc(b,3)))
        network, wave_mean, wave_std = load_lstm_info(args)
        network.to(device)  

        test_input,z_mean,z_std,roll_mean,roll_std,pitch_mean,pitch_std = load_and_simplize(simple_files, args, wave_mean=wave_mean, wave_std=wave_std)
    
        test_input    = reshape_for_time_resolution_s(test_input, args)
        # Create Dataset objects for each of our train/val/test sets

        test_dataset  = S2LDataset_s(test_input)

        # Create a PyTorch dataloader for each train/val set. Test set isn't needed until later

        test_loader   = DataLoader(test_dataset, batch_size=derived_args.test_batch_size)
        SO = test(network, device, test_loader)
        SO_lstm = reshape_full_series_s(SO,args,z_mean,z_std,roll_mean,roll_std,pitch_mean,pitch_std)
    
        for ii,filename in enumerate(save_files):
            save_data(filename, SO_lstm[ii,:,:])
    


def test(model, device, val_loader, time_res=1, return_output=True):
    '''
    Function for testing our models. One call to test() runs through every
    datapoint in our dataset once.
    model: an instance of our model
    device: either "cpu" or "cuda:0", depending on if you're running with GPU support
    val_loader: the dataloader for the data to run the model on
    '''
    # set model to evaluation mode
    model.eval()

    # we'll keep track of total loss to calculate the average later
    test_loss = 0

    if return_output:
        shape0 = 0
        for (inpt) in val_loader:
            shape0 += np.asarray(inpt).shape[0]
            shape1 = np.asarray(inpt).shape[1]
        saved_output = np.zeros((shape0, shape1, 3)) #3 = Zcg, roll, pitch
        idx0_start = 0

    #don't perform backprop if testing
    with torch.no_grad():
        # iterate thorugh each test image
        for batch_idx, (inpt) in enumerate(val_loader):
            # send inpt image to GPU if using GPU            
            inpt = inpt.to(device, dtype=torch.float)
            
            # run inpt through our model
            output = model.forward(inpt)
            idx0_end = idx0_start+inpt.size(dim=0)
            saved_output[ idx0_start:idx0_end, : ,:] = output.cpu() # For a complete end-to-end time series
            idx0_start = idx0_end
    return saved_output

def load_fullsimple(num_inputs, args, simple_filenames):
       lstm_inputs = []
       num_files = len(simple_filenames)
       num_truncate = 10 #skipping last ten-ish rows because they go to all zero in the SIMPLE file for some reason
       for k in range(num_files):
           simple_filename = simple_filenames[k]
           if args.input_option==2 or args.input_option==3:
               if args.wave_grid_x_size==1 and args.wave_grid_y_size==1:
                   wave_content = np.loadtxt(simple_filename + ".wav", skiprows=2,max_rows=18000)
                   wave_data = wave_content[:-num_truncate, 1:2]
               else:
                   wave_data = np.loadtxt(lamp_filename + ".wave_grid")[:-num_truncate, :]
           simple_content = np.loadtxt(simple_filename + ".mot",skiprows=2,max_rows=18000)
           simple_data = simple_content[:-num_truncate, [3,4,5]] # the 3,4,and 5 columns (4th, 5th, and 6th) are Zcg, roll, and Pitch                                        
           if args.input_option == 1:
               lstm_inputs.append(simple_data)
           else:
               #We have to keep the simple_data in for the moment for standardization purposes
               lstm_inputs.append(np.concatenate((simple_data, wave_data), axis=1))
       lstm_inputs = np.asarray(lstm_inputs)
       return lstm_inputs
            
def load_and_simplize(simple_filenames, args, wave_mean=None, wave_std=None):
    lstm_inputs = load_fullsimple(args.input_size, args, simple_filenames)
    num_datasets = lstm_inputs.shape[0]
    flag=False
    if wave_mean==None and wave_std==None:
        flag = True
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
        lstm_inputs[i:i+1,:,3:] = (lstm_inputs[i:i+1,:,3:]-wave_mean)/wave_std

    if args.input_option==1:
        lstm_inputs = lstm_inputs[:,:,:3]
    elif args.input_option==2:
        lstm_inputs = lstm_inputs[:,:,3:]
    print("full series simple (input) shape ", lstm_inputs.shape)
    if flag:
        return lstm_inputs, wave_mean, wave_std
    else:
        return lstm_inputs,zcg_mean,zcg_std,roll_mean,roll_std,pitch_mean,pitch_std
   
def reshape_for_time_resolution_s(lstm_inputs, args,zcg_mean,zcg_std,roll_mean,roll_std,pitch_mean,pitch_std):
    # Returns data (inputs) and labels (targets)
    data = []
    num_truncate = lstm_inputs.shape[1]%args.time_res
    end_idx = lstm_inputs.shape[1] - num_truncate
    data = lstm_inputs[:, :end_idx, :].reshape(-1, end_idx//args.time_res, args.input_size, order='F')
    data[:,:,0] = data[:,:,0]*zcg_std + zcg_mean
    data[:,:,1] = data[:,:,1]*roll_std + roll_mean
    data[:,:,2] = data[:,:,1]*pitch_std + pitch_mean
    return data

def reshape_full_series_s(lstm_outputs, args):
    num_realizations = lstm_outputs.shape[0]//args.time_res

    outputs = lstm_outputs.reshape(num_realizations, -1, 3, order='F')
    
    return outputs
class S2LDataset_s(Dataset):
    '''
    A custom dataset class to use with PyTorch's built-in dataloaders.
    This will make feeding data to our models much easier downstream.

    data: np.arrays
    '''
    def __init__(self, data, vectorize=False):
        self.data = data
    
    def __getitem__(self, idx):
        sequence_data = self.data[idx]
        return sequence_data

    def __len__(self):
        return self.data.shape[0] 
    
    


def save_data(filename, output):
	my_header = "Zcg        Roll        Pitch"
	try:
		np.savetxt(filename, output, fmt="%.5f" ,delimiter='     ', header=my_header)
		print("Successfully saved output in ", filename)
	except:
		print("Error saving file (possibly the output path). Attempting to save in current directory")
		print("output path is: ", filename)
		try:
			np.savetxt(filename, output, delimiter='     ', fmt="%.5f", header=my_header)
			print("Successfully saved output in current directory: ", filename)
		except:
			print("Failed to save file called ", filename)
    
if __name__=='__main__':
    main()