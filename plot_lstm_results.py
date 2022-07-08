import matplotlib.pyplot as plt
from peak_errors_scatter_plot import peak_errors_scatter_plot
import numpy as np
def plot_lstm_results(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, train_sc, val_sc, test_sc, plot_args, data_info_args, std_factors):
	if plot_args.plotting_mode==False:
		return

	for i in plot_args.prediction_ID_list_train:
		prediction_plot(train_target[i], train_lstm_output[i], train_sc[i], plot_args, data_info_args.train_sc[i], std_factors)

	for i in plot_args.prediction_ID_list_val:
		prediction_plot(val_target[i], val_lstm_output[i], val_sc[i], plot_args, data_info_args.val_sc[i], std_factors)

	for i in plot_args.prediction_ID_list_test:
		prediction_plot(test_target[i], test_lstm_output[i], test_sc[i], plot_args, data_info_args.test_sc[i], std_factors)

	for i in plot_args.error_ID_list_train:
		error_plot(train_target[i], train_lstm_output[i], train_sc[i], plot_args, data_info_args.train_sc[i], std_factors)

	for i in plot_args.error_ID_list_val:
		error_plot(val_target[i], val_lstm_output[i], val_sc[i], plot_args, data_info_args.val_sc[i], std_factors)

	for i in plot_args.error_ID_list_test:
		error_plot(test_target[i], test_lstm_output[i], test_sc[i], plot_args, data_info_args.test_sc[i], std_factors)

	maxima_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, train_sc, val_sc, test_sc, plot_args, data_info_args, std_factors)

	# histogram_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, train_sc, val_sc, test_sc, plot_args, data_info_args)

	plt.draw()


def prediction_plot(lamp_data, lstm_data, simple_data, plot_args, sc_filepath, std_factors):
	#un-standardize the data
	print("Plotting the parameters prediction plots for ", sc_filepath)
	realization_length = lstm_data.shape[0]
	# simple_data = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
	time_vector = [i/10 for i in range(lstm_data.shape[0])]

	fig1, axs1 = plt.subplots(3,1)
	plt.subplots_adjust(hspace=0.4)

	# Validation set, LSTM and LAMP results
	axs1[0].plot(time_vector, lamp_data[:,0], color=plot_args.lamp_color, linewidth=0.75)
	axs1[0].plot(time_vector, simple_data[:,0], ':', color=plot_args.simple_color)
	axs1[0].plot(time_vector, lstm_data[:,0], ':', color=plot_args.lstm_color)	
	axs1[0].legend(["LAMP","SIMPLE","SIMPLE + LSTM correction"])
	axs1[0].title.set_text("Zcg")
	axs1[0].set(xlabel="Time (seconds)", ylabel="Zcg (meters)")
	# axs1[0].set_xlim([1600, 1700])
	# axs1[0].set_ylim([-5, 5])
	
	axs1[1].plot(time_vector, lamp_data[:,1], color=plot_args.lamp_color, linewidth=0.75)
	axs1[1].plot(time_vector, simple_data[:,1], ':', color=plot_args.simple_color)	
	axs1[1].plot(time_vector, lstm_data[:,1], ':', color=plot_args.lstm_color)	
	axs1[1].legend(["LAMP","SIMPLE","SIMPLE + LSTM correction"])
	axs1[1].title.set_text("Roll")
	axs1[1].set(xlabel="Time (seconds)", ylabel="Roll (degrees)")
	# axs1[1].set_xlim([1600, 1700])
	# axs1[1].set_ylim([-5, 5])

	axs1[2].plot(time_vector, lamp_data[:,2], color=plot_args.lamp_color, linewidth=0.75)
	axs1[2].plot(time_vector, simple_data[:,2], ':', color=plot_args.simple_color)	
	axs1[2].plot(time_vector, lstm_data[:,2], ':', color=plot_args.lstm_color)	
	axs1[2].legend(["LAMP","SIMPLE","SIMPLE + LSTM correction"])
	axs1[2].title.set_text("Pitch")
	axs1[2].set(xlabel="Time (seconds)", ylabel="Pitch (degrees)")
	# axs1[2].set_xlim([1600, 1700])
	# axs1[2].set_ylim([-5, 5])


def error_plot(lamp_data, lstm_data, simple_data, plot_args, sc_filepath, std_factors):
	#un-standardize the data
	print("Plotting the error plots for ", sc_filepath)
	realization_length = lstm_data.shape[0]
	# simple_data = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
	time_vector = [i/10 for i in range(lstm_data.shape[0])]

	fig2, axs2 = plt.subplots(3,2)
	plt.subplots_adjust(hspace=0.5)

	lstm_error = lstm_data - lamp_data
	simple_error = simple_data - lamp_data

	# Validation set, SIMPLE error and LSTM error vs. time
	running_avg_simple_error = np.copy(simple_error)
	running_avg_lstm_error   = np.copy(lstm_error)
	span = plot_args.running_avg_span #to the left and to the right
	for i in range(lstm_data.shape[0]):
		running_avg_simple_error[i,:] = np.mean(np.abs(simple_error[max(0,i-span):min(lstm_data.shape[0],i+span),:]),axis=0)
		running_avg_lstm_error[i,:]   = np.mean(np.abs(  lstm_error[max(0,i-span):min(lstm_data.shape[0],i+span),:]),axis=0)

	axs2[0, 0].plot(time_vector, running_avg_simple_error[:,0], color=plot_args.simple_color)
	axs2[0, 0].plot(time_vector, running_avg_lstm_error[:,0], color=plot_args.lstm_color)	
	axs2[0, 0].legend(["SIMPLE","SIMPLE + LSTM correction"])
	axs2[0, 0].title.set_text("Running Avg Zcg Error\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[:,0])), np.mean(np.absolute(simple_error[:,0])), np.std(np.absolute(simple_error[:,0])),
		np.max(np.absolute(lstm_error[:,0])), np.mean(np.absolute(lstm_error[:,0])), np.std(np.absolute(lstm_error[:,0]))))
	axs2[0, 0].set(xlabel="Time (seconds)", ylabel="Model-LAMP (meters)")
	# axs2[0, 0].set_xlim([0, 1800])
	# axs2[0, 0].set_ylim([-0.1, 2])

	axs2[1, 0].plot(time_vector, running_avg_simple_error[:,1], color=plot_args.simple_color)
	axs2[1, 0].plot(time_vector, running_avg_lstm_error[:,1], color=plot_args.lstm_color)	
	axs2[1, 0].legend(["SIMPLE Error","SIMPLE + LSTM correction"])
	axs2[1, 0].title.set_text("Running Avg Roll Error\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[:,1])), np.mean(np.absolute(simple_error[:,1])), np.std(np.absolute(simple_error[:,1])),
		np.max(np.absolute(lstm_error[:,1])), np.mean(np.absolute(lstm_error[:,1])), np.std(np.absolute(lstm_error[:,1]))))
	axs2[1, 0].set(xlabel="Time (seconds)", ylabel="Model-LAMP (degrees)")
	# axs2[1, 0].set_xlim([0, 1800])
	# axs2[1, 0].set_ylim([-0.1, 2])

	axs2[2, 0].plot(time_vector, running_avg_simple_error[:,2], color=plot_args.simple_color)
	axs2[2, 0].plot(time_vector, running_avg_lstm_error[:,2], color=plot_args.lstm_color)	
	axs2[2, 0].legend(["SIMPLE Error","SIMPLE + LSTM correction"])
	axs2[2, 0].title.set_text("Running Avg Pitch Error\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[:,2])), np.mean(np.absolute(simple_error[:,2])), np.std(np.absolute(simple_error[:,2])),
		np.max(np.absolute(lstm_error[:,2])), np.mean(np.absolute(lstm_error[:,2])), np.std(np.absolute(lstm_error[:,2]))))
	axs2[2, 0].set(xlabel="Time (seconds)", ylabel="Model-LAMP (degrees)")
	# axs2[2, 0].set_xlim([0, 1800])

	# Generates scatter plot:
	# X-axis is LAMP peak values; Y-axis is correlating LSTM
	# error in one color and SIMPLE error in another color.
	zcg_peak_idx, roll_peak_idx, pitch_peak_idx = peak_errors_scatter_plot(lamp_data[:,:])
	axs2[0, 1].scatter(lamp_data[ zcg_peak_idx, 0], simple_error[ zcg_peak_idx, 0], s=1, color=plot_args.simple_color)
	axs2[0, 1].scatter(lamp_data[ zcg_peak_idx, 0], lstm_error[ zcg_peak_idx, 0], s=1, color=plot_args.lstm_color)	
	axs2[0, 1].legend(["SIMPLE","SIMPLE + LSTM correction"])
	axs2[0, 1].title.set_text("Zcg Model Errors at Peaks\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[zcg_peak_idx,0])), np.mean(np.absolute(simple_error[zcg_peak_idx,0])), np.std(np.absolute(simple_error[zcg_peak_idx,0])),
		np.max(np.absolute(lstm_error[zcg_peak_idx,0])), np.mean(np.absolute(lstm_error[zcg_peak_idx,0])), np.std(np.absolute(lstm_error[zcg_peak_idx,0]))))
	axs2[0, 1].set(xlabel="LAMP Zcg at Peak (meters)", ylabel="Model Error at LAMP Peak (meters)")
	axs2[0, 1].set_ylim([-(np.max(np.absolute(simple_error[zcg_peak_idx,0]))+0.2), np.max(np.absolute(simple_error[zcg_peak_idx,0]))+0.2])

	if not len(roll_peak_idx)==0:
		axs2[1, 1].scatter(lamp_data[ roll_peak_idx, 1], simple_error[ roll_peak_idx, 1], s=1, color=plot_args.simple_color)
		axs2[1, 1].scatter(lamp_data[ roll_peak_idx, 1], lstm_error[ roll_peak_idx, 1], s=1, color=plot_args.lstm_color)	
		axs2[1, 1].legend(["SIMPLE","SIMPLE + LSTM correction"])
		axs2[1, 1].title.set_text("Roll Model Errors at Peaks\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
			np.max(np.absolute(simple_error[roll_peak_idx,1])), np.mean(np.absolute(simple_error[roll_peak_idx,1])), np.std(np.absolute(simple_error[roll_peak_idx,1])),
			np.max(np.absolute(lstm_error[roll_peak_idx,1])), np.mean(np.absolute(lstm_error[roll_peak_idx,1])), np.std(np.absolute(lstm_error[roll_peak_idx,1]))))
		axs2[1, 1].set(xlabel="LAMP Roll at Peak (degrees)", ylabel="Model Error at LAMP Peak (degrees)")
		axs2[1, 1].set_ylim([-(np.max(np.absolute(simple_error[roll_peak_idx,1]))+0.2), np.max(np.absolute(simple_error[roll_peak_idx,1]))+0.2])

	axs2[2, 1].scatter(lamp_data[ pitch_peak_idx, 2], simple_error[ pitch_peak_idx, 2], s=1, color=plot_args.simple_color)
	axs2[2, 1].scatter(lamp_data[ pitch_peak_idx, 2], lstm_error[ pitch_peak_idx, 2], s=1, color=plot_args.lstm_color)	
	axs2[2, 1].legend(["SIMPLE","SIMPLE + LSTM correction"])
	axs2[2, 1].title.set_text("Pitch Model Errors at Peaks\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[pitch_peak_idx,2])), np.mean(np.absolute(simple_error[pitch_peak_idx,2])), np.std(np.absolute(simple_error[pitch_peak_idx,2])),
		np.max(np.absolute(lstm_error[pitch_peak_idx,2])), np.mean(np.absolute(lstm_error[pitch_peak_idx,2])), np.std(np.absolute(lstm_error[pitch_peak_idx,2]))))
	axs2[2, 1].set(xlabel="LAMP Pitch at Peak (degrees)", ylabel="Model Error at LAMP Peak (degrees)")
	axs2[2, 1].set_ylim([-(np.max(np.absolute(simple_error[pitch_peak_idx,2]))+0.2), np.max(np.absolute(simple_error[pitch_peak_idx,2]))+0.2])


def maxima_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, train_simple_data, val_simple_data, test_simple_data, plot_args, data_info_args, std_factors):
	num_realizations = len(plot_args.maxima_ID_list_train) + len(plot_args.maxima_ID_list_val) + len(plot_args.maxima_ID_list_test)
	if num_realizations==0:
		return
	print("Plotting the maxima plots")
	simple_data = np.zeros((num_realizations, train_target.shape[1], 3))
	lamp_data   = np.zeros((num_realizations, train_target.shape[1], 3))
	lstm_data   = np.zeros((num_realizations, train_target.shape[1], 3))
	j = 0
	for i in plot_args.maxima_ID_list_train:
		sc_filepath = data_info_args.train_sc[i]
		# realization_length = lstm_data.shape[1]
		# simple_data[j] = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
		simple_data[j] = np.copy(train_simple_data[i])
		lamp_data[j] = np.copy(train_target[i])
		lstm_data[j] = np.copy(train_lstm_output[i])
		j+=1
	j1 = j

	for i in plot_args.maxima_ID_list_val:
		sc_filepath = data_info_args.val_sc[i]
		# realization_length = lstm_data.shape[1]
		# simple_data[j] = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
		simple_data[j] = np.copy(val_simple_data[i])
		lamp_data[j] = np.copy(val_target[i])
		lstm_data[j] = np.copy(val_lstm_output[i])
		j+=1
	j2 = j

	for i in plot_args.maxima_ID_list_test:
		sc_filepath = data_info_args.test_sc[i]
		# realization_length = lstm_data.shape[1]
		# simple_data[j] = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
		simple_data[j] = np.copy(test_simple_data[i])
		lamp_data[j] = np.copy(test_target[i])
		lstm_data[j] = np.copy(test_lstm_output[i])
		j+=1
	j3 = j

	# time_vector = [i/10 for i in range(lstm_data.shape[1])]

	max_zcg_lamp 		= np.max(np.abs(lamp_data[:,:,0]), axis=1)
	max_roll_lamp 		= np.max(np.abs(lamp_data[:,:,1]), axis=1)
	max_pitch_lamp 		= np.max(np.abs(lamp_data[:,:,2]), axis=1)
	max_zcg_simple 		= np.max(np.abs(simple_data[:,:,0]), axis=1)
	max_roll_simple 	= np.max(np.abs(simple_data[:,:,1]), axis=1)
	max_pitch_simple 	= np.max(np.abs(simple_data[:,:,2]), axis=1)
	max_zcg_lstm 		= np.max(np.abs(lstm_data[:,:,0]), axis=1)
	max_roll_lstm 		= np.max(np.abs(lstm_data[:,:,1]), axis=1)
	max_pitch_lstm 		= np.max(np.abs(lstm_data[:,:,2]), axis=1)

	fig3, axs3 = plt.subplots(3,1)
	bullet_size = plot_args.maxima_bullet_size
	plt.subplots_adjust(hspace=0.4)
	axs3[0].plot([0,20],[0,20], color="black", linestyle='dashed')
	axs3[1].plot([0,100],[0,100], color="black", linestyle='dashed')
	axs3[2].plot([0,20],[0,20], color="black", linestyle='dashed')
	axs3[0].scatter(max_zcg_lamp[:j1], max_zcg_simple[:j1], s=bullet_size, color=plot_args.simple_color, marker="x")
	axs3[0].scatter(max_zcg_lamp[:j1], max_zcg_lstm[:j1], s=bullet_size, color=plot_args.lstm_color, marker="x")
	axs3[0].scatter(max_zcg_lamp[j1:j2], max_zcg_simple[j1:j2], s=bullet_size, color=plot_args.simple_color, marker="D")
	axs3[0].scatter(max_zcg_lamp[j1:j2], max_zcg_lstm[j1:j2], s=bullet_size, color=plot_args.lstm_color, marker="D")
	axs3[0].scatter(max_zcg_lamp[j2:j3], max_zcg_simple[j2:j3], s=bullet_size, color=plot_args.simple_color, marker="*")
	axs3[0].scatter(max_zcg_lamp[j2:j3], max_zcg_lstm[j2:j3], s=bullet_size, color=plot_args.lstm_color, marker="*")

	axs3[1].scatter(max_roll_lamp[:j1], max_roll_simple[:j1], s=bullet_size, color=plot_args.simple_color, marker="x")
	axs3[1].scatter(max_roll_lamp[:j1], max_roll_lstm[:j1], s=bullet_size, color=plot_args.lstm_color, marker="x")
	axs3[1].scatter(max_roll_lamp[j1:j2], max_roll_simple[j1:j2], s=bullet_size, color=plot_args.simple_color, marker="D")
	axs3[1].scatter(max_roll_lamp[j1:j2], max_roll_lstm[j1:j2], s=bullet_size, color=plot_args.lstm_color, marker="D")
	axs3[1].scatter(max_roll_lamp[j2:j3], max_roll_simple[j2:j3], s=bullet_size, color=plot_args.simple_color, marker="*")
	axs3[1].scatter(max_roll_lamp[j2:j3], max_roll_lstm[j2:j3], s=bullet_size, color=plot_args.lstm_color, marker="*")

	axs3[2].scatter(max_pitch_lamp[:j1], max_pitch_simple[:j1], s=bullet_size, color=plot_args.simple_color, marker="x")
	axs3[2].scatter(max_pitch_lamp[:j1], max_pitch_lstm[:j1], s=bullet_size, color=plot_args.lstm_color, marker="x")
	axs3[2].scatter(max_pitch_lamp[j1:j2], max_pitch_simple[j1:j2], s=bullet_size, color=plot_args.simple_color, marker="D")
	axs3[2].scatter(max_pitch_lamp[j1:j2], max_pitch_lstm[j1:j2], s=bullet_size, color=plot_args.lstm_color, marker="D")
	axs3[2].scatter(max_pitch_lamp[j2:j3], max_pitch_simple[j2:j3], s=bullet_size, color=plot_args.simple_color, marker="*")
	axs3[2].scatter(max_pitch_lamp[j2:j3], max_pitch_lstm[j2:j3], s=bullet_size, color=plot_args.lstm_color, marker="*")

	axs3[0].title.set_text("Zcg Absolute Value Maxima per Realization")
	axs3[1].title.set_text("Roll Absolute Value Maxima per Realization")
	axs3[2].title.set_text("Pitch Absolute Value Maxima per Realization")
	axs3[0].legend(["Perfect","SC Training", "LSTM Training", "SC Val", "LSTM Val", "SC Test", "LSTM Test"], bbox_to_anchor=(1.1, 1.05))
	axs3[1].legend(["Perfect","SC Training", "LSTM Training", "SC Val", "LSTM Val", "SC Test", "LSTM Test"], bbox_to_anchor=(1.1, 1.05))
	axs3[2].legend(["Perfect","SC Training", "LSTM Training", "SC Val", "LSTM Val", "SC Test", "LSTM Test"], bbox_to_anchor=(1.1, 1.05))
	axs3[0].set(xlabel="LAMP Zcg Maxima (meters)", ylabel="Corresponding Maxima from SC or LSTM")
	axs3[1].set(xlabel="LAMP Roll Maxima (degrees)", ylabel="Corresponding Maxima from SC or LSTM")
	axs3[2].set(xlabel="LAMP Pitch Maxima (degrees)", ylabel="Corresponding Maxima from SC or LSTM")
	axs3[0].set_xlim([np.min((max_zcg_lamp, max_zcg_simple))-1, np.max((max_zcg_lamp, max_zcg_simple))+1])
	axs3[1].set_xlim([np.min((max_roll_lamp, max_roll_simple))-1, np.max((max_roll_lamp, max_roll_simple))+1])
	axs3[2].set_xlim([np.min((max_pitch_lamp, max_pitch_simple))-1, np.max((max_pitch_lamp, max_pitch_simple))+1])
	axs3[0].set_ylim([np.min((max_zcg_lamp, max_zcg_simple))-1, np.max((max_zcg_lamp, max_zcg_simple))+1])
	axs3[1].set_ylim([np.min((max_roll_lamp, max_roll_simple))-1, np.max((max_roll_lamp, max_roll_simple))+1])
	axs3[2].set_ylim([np.min((max_pitch_lamp, max_pitch_simple))-1, np.max((max_pitch_lamp, max_pitch_simple))+1])
	axs3[0].axes.set_aspect('equal')
	axs3[1].axes.set_aspect('equal')
	axs3[2].axes.set_aspect('equal')
	
def add_to_histo_list(data, zcg_list, roll_list, pitch_list):
	zcg_peak_idx, roll_peak_idx, pitch_peak_idx = peak_errors_scatter_plot(data[600:,:])
	zcg_list.extend(data[zcg_peak_idx,0])
	roll_list.extend(data[roll_peak_idx,1])
	pitch_list.extend(data[pitch_peak_idx,2])

	# zcg_list.extend(data[:,0])
	# roll_list.extend(data[:,1])
	# pitch_list.extend(data[:,2])

def histogram_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, train_simple_data, val_simple_data, test_simple_data, plot_args, data_info_args):
	lamp_zcg_histo_list = []
	lamp_roll_histo_list = []
	lamp_pitch_histo_list = []
	sc_zcg_histo_list = []
	sc_roll_histo_list = []
	sc_pitch_histo_list = []
	lstm_zcg_histo_list = []
	lstm_roll_histo_list = []
	lstm_pitch_histo_list = []
	
	print("Plotting the histogram plot for")
	for i in plot_args.histo_ID_list_train:
		print(data_info_args.train_sc[i])
		add_to_histo_list(train_target[i], 			lamp_zcg_histo_list, lamp_roll_histo_list, lamp_pitch_histo_list)
		add_to_histo_list(train_simple_data[i],   	sc_zcg_histo_list,   sc_roll_histo_list,   sc_pitch_histo_list)
		add_to_histo_list(train_lstm_output[i], 	lstm_zcg_histo_list, lstm_roll_histo_list, lstm_pitch_histo_list)
		
	for i in plot_args.histo_ID_list_val:
		print(data_info_args.val_sc[i])
		add_to_histo_list(val_target[i], 		lamp_zcg_histo_list, lamp_roll_histo_list, lamp_pitch_histo_list)
		add_to_histo_list(val_simple_data[i],   sc_zcg_histo_list,   sc_roll_histo_list,   sc_pitch_histo_list)
		add_to_histo_list(val_lstm_output[i], 	lstm_zcg_histo_list, lstm_roll_histo_list, lstm_pitch_histo_list)

	for i in plot_args.histo_ID_list_test:
		print(data_info_args.test_sc[i])
		add_to_histo_list(test_target[i], 		lamp_zcg_histo_list, lamp_roll_histo_list, lamp_pitch_histo_list)
		add_to_histo_list(test_simple_data[i], 	sc_zcg_histo_list,   sc_roll_histo_list,   sc_pitch_histo_list)
		add_to_histo_list(test_lstm_output[i], 	lstm_zcg_histo_list, lstm_roll_histo_list, lstm_pitch_histo_list)

	zcg_min = min(sc_zcg_histo_list + lamp_zcg_histo_list + lstm_zcg_histo_list)
	zcg_max = max(sc_zcg_histo_list + lamp_zcg_histo_list + lstm_zcg_histo_list)
	roll_min = min(sc_roll_histo_list + lamp_roll_histo_list + lstm_roll_histo_list)
	roll_max = max(sc_roll_histo_list + lamp_roll_histo_list + lstm_roll_histo_list)
	pitch_min = min(sc_pitch_histo_list + lamp_pitch_histo_list + lstm_pitch_histo_list)
	pitch_max = max(sc_pitch_histo_list  + lamp_pitch_histo_list + lstm_pitch_histo_list)
	n_bins = 20
	fig4, axs4 = plt.subplots(3, 3, sharey='row', tight_layout=True)
	density_bool = False
	axs4[0,0].hist(sc_zcg_histo_list, bins=n_bins, color=plot_args.simple_color, range=(zcg_min, zcg_max), density=density_bool)
	axs4[0,1].hist(lamp_zcg_histo_list, bins=n_bins, color=plot_args.lamp_color, range=(zcg_min, zcg_max), density=density_bool)
	axs4[0,2].hist(lstm_zcg_histo_list, bins=n_bins, color=plot_args.lstm_color, range=(zcg_min, zcg_max), density=density_bool)
	axs4[1,0].hist(sc_roll_histo_list, bins=n_bins, color=plot_args.simple_color, range=(roll_min, roll_max), density=density_bool)
	axs4[1,1].hist(lamp_roll_histo_list, bins=n_bins, color=plot_args.lamp_color, range=(roll_min, roll_max), density=density_bool)
	axs4[1,2].hist(lstm_roll_histo_list, bins=n_bins, color=plot_args.lstm_color, range=(roll_min, roll_max), density=density_bool)
	axs4[2,0].hist(sc_pitch_histo_list, bins=n_bins, color=plot_args.simple_color, range=(pitch_min, pitch_max), density=density_bool)
	axs4[2,1].hist(lamp_pitch_histo_list, bins=n_bins, color=plot_args.lamp_color, range=(pitch_min, pitch_max), density=density_bool)
	axs4[2,2].hist(lstm_pitch_histo_list, bins=n_bins, color=plot_args.lstm_color, range=(pitch_min, pitch_max), density=density_bool)



