import matplotlib.pyplot as plt
from peak_errors_scatter_plot import peak_errors_scatter_plot
import numpy as np
def plot_lstm_results(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, plot_args, data_info_args):
	if plot_args.plotting_mode==False:
		return

	for i in plot_args.prediction_ID_list_train:
		prediction_plot(train_target[i], train_lstm_output[i], plot_args, data_info_args.train_sc[i])

	for i in plot_args.prediction_ID_list_val:
		prediction_plot(val_target[i], val_lstm_output[i], plot_args, data_info_args.val_sc[i])

	for i in plot_args.prediction_ID_list_test:
		prediction_plot(test_target[i], test_lstm_output[i], plot_args, data_info_args.test_sc[i])

	for i in plot_args.error_ID_list_train:
		error_plot(train_target[i], train_lstm_output[i], plot_args, data_info_args.train_sc[i])

	for i in plot_args.error_ID_list_val:
		error_plot(val_target[i], val_lstm_output[i], plot_args, data_info_args.val_sc[i])

	for i in plot_args.error_ID_list_test:
		error_plot(test_target[i], test_lstm_output[i], plot_args, data_info_args.test_sc[i])

	maxima_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, plot_args, data_info_args)
	plt.draw()


def prediction_plot(target, lstm_output, plot_args, sc_filepath):
	#un-standardize the data
	print("Plotting the parameters prediction plots for ", sc_filepath)
	simple_data, lamp_data, lstm_data = un_standardize(target, lstm_output, sc_filepath)
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
	axs1[0].set_xlim([1600, 1700])
	# axs1[0].set_ylim([-5, 5])
	
	axs1[1].plot(time_vector, lamp_data[:,1], color=plot_args.lamp_color, linewidth=0.75)
	axs1[1].plot(time_vector, simple_data[:,1], ':', color=plot_args.simple_color)	
	axs1[1].plot(time_vector, lstm_data[:,1], ':', color=plot_args.lstm_color)	
	axs1[1].legend(["LAMP","SIMPLE","SIMPLE + LSTM correction"])
	axs1[1].title.set_text("Roll")
	axs1[1].set(xlabel="Time (seconds)", ylabel="Roll (degrees)")
	axs1[1].set_xlim([1600, 1700])
	# axs1[1].set_ylim([-5, 5])

	axs1[2].plot(time_vector, lamp_data[:,2], color=plot_args.lamp_color, linewidth=0.75)
	axs1[2].plot(time_vector, simple_data[:,2], ':', color=plot_args.simple_color)	
	axs1[2].plot(time_vector, lstm_data[:,2], ':', color=plot_args.lstm_color)	
	axs1[2].legend(["LAMP","SIMPLE","SIMPLE + LSTM correction"])
	axs1[2].title.set_text("Pitch")
	axs1[2].set(xlabel="Time (seconds)", ylabel="Pitch (degrees)")
	axs1[2].set_xlim([1600, 1700])
	# axs1[2].set_ylim([-5, 5])


def error_plot(target, lstm_output, plot_args, sc_filepath):
	#un-standardize the data
	print("Plotting the error plots for ", sc_filepath)
	simple_data, lamp_data, lstm_data = un_standardize(target, lstm_output, sc_filepath)
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
	axs2[0, 0].set_xlim([0, 1800])
	# axs2[0, 0].set_ylim([-0.1, 2])

	axs2[1, 0].plot(time_vector, running_avg_simple_error[:,1], color=plot_args.simple_color)
	axs2[1, 0].plot(time_vector, running_avg_lstm_error[:,1], color=plot_args.lstm_color)	
	axs2[1, 0].legend(["SIMPLE Error","SIMPLE + LSTM correction"])
	axs2[1, 0].title.set_text("Running Avg Roll Error\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[:,1])), np.mean(np.absolute(simple_error[:,1])), np.std(np.absolute(simple_error[:,1])),
		np.max(np.absolute(lstm_error[:,1])), np.mean(np.absolute(lstm_error[:,1])), np.std(np.absolute(lstm_error[:,1]))))
	axs2[1, 0].set(xlabel="Time (seconds)", ylabel="Model-LAMP (degrees)")
	axs2[1, 0].set_xlim([0, 1800])
	# axs2[1, 0].set_ylim([-0.1, 2])

	axs2[2, 0].plot(time_vector, running_avg_simple_error[:,2], color=plot_args.simple_color)
	axs2[2, 0].plot(time_vector, running_avg_lstm_error[:,2], color=plot_args.lstm_color)	
	axs2[2, 0].legend(["SIMPLE Error","SIMPLE + LSTM correction"])
	axs2[2, 0].title.set_text("Running Avg Pitch Error\n SIMPLE Abolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f} \n LSTM Absolute Error Max={:.3f},    Mean={:.3f},    STD={:.3f}".format(
		np.max(np.absolute(simple_error[:,2])), np.mean(np.absolute(simple_error[:,2])), np.std(np.absolute(simple_error[:,2])),
		np.max(np.absolute(lstm_error[:,2])), np.mean(np.absolute(lstm_error[:,2])), np.std(np.absolute(lstm_error[:,2]))))
	axs2[2, 0].set(xlabel="Time (seconds)", ylabel="Model-LAMP (degrees)")
	axs2[2, 0].set_xlim([0, 1800])

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


def maxima_plot(train_target, val_target, test_target, train_lstm_output, val_lstm_output, test_lstm_output, plot_args, data_info_args):
	#un-standardize the data
	
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
		simple_data[j], lamp_data[j], lstm_data[j] = un_standardize(train_target[i], train_lstm_output[i], sc_filepath)
		j+=1
	j1 = j

	for i in plot_args.maxima_ID_list_val:
		sc_filepath = data_info_args.val_sc[i]
		simple_data[j], lamp_data[j], lstm_data[j] = un_standardize(val_target[i], val_lstm_output[i], sc_filepath)
		j+=1
	j2 = j

	for i in plot_args.maxima_ID_list_test:
		sc_filepath = data_info_args.test_sc[i]
		simple_data[j], lamp_data[j], lstm_data[j] = un_standardize(test_target[i], test_lstm_output[i], sc_filepath)
		j+=1
	j3 = j

	time_vector = [i/10 for i in range(lstm_data.shape[1])]

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


def un_standardize(target, lstm_output, sc_filepath):
	realization_length = lstm_output.shape[0]
	simple_data = np.loadtxt(sc_filepath + ".mot",skiprows=2)[:realization_length,[3,4,5]]
	zcg_mean = np.mean(simple_data[:,0])
	zcg_std = np.std(simple_data[:,0])
	roll_mean = np.mean(simple_data[:,1])
	roll_std = np.std(simple_data[:,1])
	pitch_mean = np.mean(simple_data[:,2])
	pitch_std = np.std(simple_data[:,2])

	lamp_data = np.copy(target)
	lstm_data = np.copy(lstm_output)
	lamp_data[:,0] = target[:,0]*zcg_std + zcg_mean
	lamp_data[:,1] = target[:,1]*roll_std + roll_mean
	lamp_data[:,2] = target[:,2]*pitch_std + pitch_mean
	lstm_data[:,0] = lstm_output[:,0]*zcg_std + zcg_mean
	lstm_data[:,1] = lstm_output[:,1]*roll_std + roll_mean
	lstm_data[:,2] = lstm_output[:,2]*pitch_std + pitch_mean

	return simple_data, lamp_data, lstm_data
	

	
	
	

	



