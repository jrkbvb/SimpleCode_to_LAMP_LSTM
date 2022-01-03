import numpy as np 
def unstandardize_all_data(sc, target, lstm, std_factors, sc_filepath, lamp_filepath):
	num_realizations = lstm.shape[0]
	if not num_realizations==len(sc_filepath):
		print("\n\nERROR in number of realizations! See unstandardize_all_data function\n\n")
	for i in range(num_realizations):
		sc[i,:,0] 		= sc[i,:,0]		*std_factors[1] + std_factors[0]
		target[i,:,0] 	= target[i,:,0]	*std_factors[1] + std_factors[0]
		lstm[i,:,0] 	= lstm[i,:,0]	*std_factors[1] + std_factors[0] 

	
		sc[i,:,1] 		= sc[i,:,1]		*std_factors[3] + std_factors[2]
		target[i,:,1] 	= target[i,:,1]	*std_factors[3] + std_factors[2]
		lstm[i,:,1] 	= lstm[i,:,1]	*std_factors[3] + std_factors[2] 

	
		sc[i,:,2] 		= sc[i,:,2]		*std_factors[5] + std_factors[4]
		target[i,:,2] 	= target[i,:,2]	*std_factors[5] + std_factors[4]
		lstm[i,:,2] 	= lstm[i,:,2]	*std_factors[5] + std_factors[4] 		

	return sc, target, lstm