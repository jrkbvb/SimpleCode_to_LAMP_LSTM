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
    # inpt: a pytorch tensor representing a batch of inpt sequences.
    for batch_idx, (inpt,target) in enumerate(train_loader):
        # This line sends data to GPU if you're using a GPU
        inpt = inpt.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        # initialze the optimizer
        optimizer.zero_grad()

        # feed our inpt through the network
        output = model.forward(inpt)

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
        shape0 = 0
        for (inpt,target) in val_loader:
            shape0 += np.asarray(target).shape[0]
            shape1 = np.asarray(target).shape[1]
        saved_output = np.zeros((shape0, shape1, 3)) #3 = Zcg, roll, pitch
        idx0_start = 0

    #don't perform backprop if testing
    with torch.no_grad():
        # iterate thorugh each test image
        for batch_idx, (inpt,target) in enumerate(val_loader):
            # send inpt image to GPU if using GPU            
            inpt = inpt.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            # run inpt through our model
            output = model.forward(inpt)
            loss_function = objective_function
            loss_value = loss_function(output, target, obj_fun_hyp) 
            test_loss += loss_value

            if return_output:
                idx0_end = idx0_start+inpt.size(dim=0)
                saved_output[ idx0_start:idx0_end, : ,:] = output.cpu() # For a complete end-to-end time series
                idx0_start = idx0_end
    test_loss /= len(val_loader)
    if return_output:
    	return saved_output
    else:
        return test_loss.item()