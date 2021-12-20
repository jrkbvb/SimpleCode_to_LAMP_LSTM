import numpy as np 
def unstandardize_all_data(sc, target, lstm, std_factors, sc_filepath, lamp_filepath):
	std_type = [0, 0, 0, 0]

	num_realizations = lstm.shape[0]
	realization_length = lstm.shape[1]
	if not num_realizations==len(sc_filepath):
		print("\n\nERROR in number of realizations! See unstandardize_all_data function\n\n")
	for i in range(num_realizations):
		simple_data = np.loadtxt(sc_filepath[i] + ".mot",skiprows=2)[:realization_length,[3,4,5]]
		wave_content = np.loadtxt(lamp_filepath[i] + ".wav", skiprows=3)
		wave_data = wave_content[:realization_length, 1:2]

		zcg_mean = np.mean(simple_data[:,0])
		zcg_std = np.std(simple_data[:,0])
		roll_mean = np.mean(simple_data[:,1])
		roll_std = np.std(simple_data[:,1])
		pitch_mean = np.mean(simple_data[:,2])
		pitch_std = np.std(simple_data[:,2])
		wave_mean = np.mean(wave_data)
		wave_std = np.std(wave_data)

		if std_type[0]==0:
			sc[i,:,0] 		= sc[i,:,0]		*std_factors[1] + std_factors[0]
			target[i,:,0] 	= target[i,:,0]	*std_factors[1] + std_factors[0]
			lstm[i,:,0] 	= lstm[i,:,0]	*std_factors[1] + std_factors[0] 
		else:
			sc[i,:,0] 		= sc[i,:,0]		*zcg_std + zcg_mean
			target[i,:,0] 	= target[i,:,0]	*zcg_std + zcg_mean
			lstm[i,:,0] 	= lstm[i,:,0]	*zcg_std + zcg_mean

		if std_type[1]==0:
			sc[i,:,1] 		= sc[i,:,1]		*std_factors[3] + std_factors[2]
			target[i,:,1] 	= target[i,:,1]	*std_factors[3] + std_factors[2]
			lstm[i,:,1] 	= lstm[i,:,1]	*std_factors[3] + std_factors[2] 
		else:
			sc[i,:,1] 		= sc[i,:,1]		*roll_std + roll_mean
			target[i,:,1] 	= target[i,:,1]	*roll_std + roll_mean
			lstm[i,:,1] 	= lstm[i,:,1]	*roll_std + roll_mean

		if std_type[2]==0:
			sc[i,:,2] 		= sc[i,:,2]		*std_factors[5] + std_factors[4]
			target[i,:,2] 	= target[i,:,2]	*std_factors[5] + std_factors[4]
			lstm[i,:,2] 	= lstm[i,:,2]	*std_factors[5] + std_factors[4] 
		else:
			sc[i,:,2] 		= sc[i,:,2]		*pitch_std + pitch_mean
			target[i,:,2] 	= target[i,:,2]	*pitch_std + pitch_mean
			lstm[i,:,2] 	= lstm[i,:,2]	*pitch_std + pitch_mean
		

	return sc, target, lstm