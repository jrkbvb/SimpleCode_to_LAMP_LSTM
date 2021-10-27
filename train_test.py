import torch
from torch import nn
import numpy as np
from objective_functions import objective_function

def train(model, device, train_loader, optimizer, obj_fun_hyp):
    '''
    Function for training our networks. One call to train() performs a single
    epoch for training.
    model: an instance of our model
    device: either "cpu" or "cuda", depending on if you're running with GPU support
    train_loader: the dataloader for the training set
    optimizer: optimizer used for training (the optimizer implements SGD)
    '''

    # Set the model to training mode.
    model.train()

    #we'll keep adding the loss of each batch to total_loss, so we can calculate
    #the average loss at the end of the epoch.
    total_loss = 0
    zcg_total_loss = 0
    pitch_total_loss = 0

    # We'll iterate through each batch. One call of train() trains for 1 epoch.
    # batch_idx: an integer representing which batch number we're on
    # input: a pytorch tensor representing a batch of input sequences.
    for batch_idx, (input,target) in enumerate(train_loader):
        # This line sends data to GPU if you're using a GPU
        input = input.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        # initialze the optimizer
        optimizer.zero_grad()

        # feed our input through the network
        output = model.forward(input)

        #calculate the loss
        loss_function = objective_function
        loss_value = loss_function(output, target, obj_fun_hyp)

        # Perform backprop
        loss_value.backward()
        optimizer.step()

        #accumulate loss to later calculate the average
        total_loss += loss_value
    total_loss /= len(train_loader)
    return total_loss.item()

def test(model, device, val_loader, obj_fun_hyp, num_realizations=0, time_res=1, return_output=False):
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
    	for (input,target) in val_loader:
    		realization_length = np.asarray(target).shape[1]*time_res
    		break
    	saved_output = np.zeros((num_realizations, realization_length, 3)) #3 = Zcg, roll, pitch

    #don't perform backprop if testing
    with torch.no_grad():
        # iterate thorugh each test image
        for batch_idx, (input,target) in enumerate(val_loader):
            # send input image to GPU if using GPU
            
            input = input.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            # run input through our model
            output = model.forward(input)
            loss_function = objective_function
            loss_value = loss_function(output, target, obj_fun_hyp) 
            test_loss += loss_value

            if return_output:
            	saved_output[ batch_idx%num_realizations, batch_idx//num_realizations::time_res ,:] = output.cpu() # For a complete end-to-end time series
    test_loss /= len(val_loader)
    if return_output:
    	return saved_output
    else:
        return test_loss.item()