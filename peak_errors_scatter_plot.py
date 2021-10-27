import numpy as np
import matplotlib.pyplot as plt
def peak_errors_scatter_plot(lamp_data):
	# input: a single realization for LAMP, LSTM, and SIMPLE time series data
	# of shape [num_points in series, 3] (the 3 is for zcg, roll, and pitch)
	
	zcg_peak_idx = [] #keep track of indeces where LAMP peaks occur
	roll_peak_idx = []
	pitch_peak_idx = [] 
	num_points = lamp_data.shape[0]
	for idx in range(1,num_points-1):
		if (((lamp_data[idx,0] > lamp_data[idx+1,0]) and (lamp_data[idx,0] > lamp_data[idx-1,0])) or
			((lamp_data[idx,0] < lamp_data[idx+1,0]) and (lamp_data[idx,0] < lamp_data[idx-1,0]))):
			zcg_peak_idx.append(idx)
		if (((lamp_data[idx,1] > lamp_data[idx+1,1]) and (lamp_data[idx,1] > lamp_data[idx-1,1])) or
			((lamp_data[idx,1] < lamp_data[idx+1,1]) and (lamp_data[idx,1] < lamp_data[idx-1,1]))):
			roll_peak_idx.append(idx)
		if (((lamp_data[idx,2] > lamp_data[idx+1,2]) and (lamp_data[idx,2] > lamp_data[idx-1,2])) or
			((lamp_data[idx,2] < lamp_data[idx+1,2]) and (lamp_data[idx,2] < lamp_data[idx-1,2]))):
			pitch_peak_idx.append(idx)
	return zcg_peak_idx, roll_peak_idx, pitch_peak_idx