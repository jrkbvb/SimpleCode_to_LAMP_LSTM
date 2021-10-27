import numpy as np
import torch
from peak_errors_scatter_plot import peak_errors_scatter_plot

def objective_function(output, target, hyper_params):
	''' hyper_params[0] = number to indicate which loss/objective function will be used.
	all other indices are particular to the function.'''

	if hyper_params[0] == 0: # no modification to MSELoss, just use regular MSELoss
		loss = torch.mean((output - target)**2)
	elif hyper_params[0]==1: # MSELoss magnified by larger amplitudes
		# hyper_params[1] and [2] are for scaling the amplification. Try out different combinations
		loss = torch.mean(((output - target)**2)*(hyper_params[1] + hyper_params[2]*(target**2)))
	elif hyper_params[0]==2: # Peak MSELoss: use only the times of peaks.
		peak_idx = torch.nonzero(torch.diff(torch.sign(torch.diff(target, axis=1)), axis=1))
		peak_idx[:,1]+=1 # because of how torch.diff reduces the index
		loss = torch.mean( (target[peak_idx[:,0], peak_idx[:,1], peak_idx[:,2]] - output[peak_idx[:,0], peak_idx[:,1], peak_idx[:,2]])**2 )
	elif hyper_params[0]==3: # Extremes MSELoss. Only count errors when target is above a certain threshold
		threshold = hyper_params[1]
		threshold_idx = abs(target)>=threshold
		loss = torch.mean((output[threshold_idx]-target[threshold_idx])**2)
	elif hyper_params[0]==4: # Greedy errors MSELoss. Only count loss on large enough errors
		threshold = hyper_params[1]
		threshold_idx = abs(output - target)>=threshold
		loss = torch.mean((output[threshold_idx]-target[threshold_idx])**2)
	elif hyper_params[0]==5: # Greedy errors AND Extreme targets
		error_threshold = hyper_params[1]
		target_threshold = hyper_params[2]
		threshold_idx = torch.logical_and(abs(output - target)>=error_threshold, abs(target)>=target_threshold)
		loss = torch.mean((output[threshold_idx]-target[threshold_idx])**2)
	elif hyper_params[0]==6: # MSE, but just based off the last output of the series
		loss = torch.mean((output[:,-1,:] - target[:,-1,:])**2)

	return loss
