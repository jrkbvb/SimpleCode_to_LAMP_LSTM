import numpy as np
import matplotlib.pyplot as plt


def load_fullseries(args, simple_filenames, lamp_filenames):
    lstm_inputs = []
    target_outputs = []
    num_files = len(simple_filenames)
    num_truncate = 10  # skipping last ten-ish rows because they go to all zero in the SIMPLE file for some reason
    for k in range(num_files):
        simple_filename = simple_filenames[k]
        lamp_filename = lamp_filenames[k]
        if args.wave_grid_x_size == 1 and args.wave_grid_y_size == 1:
            wave_content = np.loadtxt(lamp_filename + ".wav", skiprows=3)
            wave_data = wave_content[:-num_truncate, 1:2]
        else:
            wave_data = np.loadtxt(lamp_filename + ".wave_grid")[:-num_truncate, :]

        simple_motion_content = np.loadtxt(simple_filename + ".mot", skiprows=2)
        lamp_motion_content = np.loadtxt(lamp_filename + ".mot", skiprows=3)
        simple_vbm_content = np.loadtxt(simple_filename + ".vbm", skiprows=2)
        lamp_vbm_content = np.loadtxt(lamp_filename + ".vbm", skiprows=4)

        simple_motion_data = simple_motion_content[
            :-num_truncate, [3, 4, 5]
        ]  # the 3,4,and 5 columns (4th, 5th, and 6th) are Zcg, roll, and Pitch
        lamp_motion_data = lamp_motion_content[:-num_truncate, [3, 4, 5]]
        simple_vbm_data = simple_vbm_content[:-num_truncate, 2:3]
        lamp_vbm_data = lamp_vbm_content[:-num_truncate, 2:3]

        lstm_inputs.append(
            np.concatenate((simple_vbm_data, simple_motion_data, wave_data), axis=1)
        )
        target_outputs.append(np.concatenate((lamp_vbm_data, lamp_motion_data), axis=1))
    lstm_inputs = np.asarray(lstm_inputs)
    target_outputs = np.asarray(target_outputs)
    return lstm_inputs, target_outputs


def load_and_standardize(simple_filenames, lamp_filenames, args, std_factors=[]):
    # lstm_inputs and target_ouputs have 3 dimensions. 1st = record number; 2nd = time index; 3rd = channel
    lstm_inputs, target_outputs = load_fullseries(
        args, simple_filenames, lamp_filenames
    )

    sc_indexing = [
        args.input_vbm,
        args.input_3dof,
        args.input_3dof,
        args.input_3dof,
        False,
    ]  # exclude waves
    input_indexing = [
        args.input_vbm,
        args.input_3dof,
        args.input_3dof,
        args.input_3dof,
        args.input_waves,
    ]
    output_indexing = [
        args.output_vbm,
        args.output_3dof,
        args.output_3dof,
        args.output_3dof,
    ]
    sc_inputs = np.copy(lstm_inputs[:, :, sc_indexing])  # exclude waves
    lstm_inputs = lstm_inputs[:, :, input_indexing]
    target_outputs = target_outputs[:, :, output_indexing]
    print("full series simple (input) shape ", lstm_inputs.shape)
    print("full series lamp (target) shape ", target_outputs.shape)

    num_datasets = lstm_inputs.shape[0]
    flag = False
    # get standardization factors if we don't have them already
    if len(std_factors) == 0:
        flag = True
        for i in range(args.input_size):
            my_mean = np.mean(lstm_inputs[:, :, i])
            my_std = np.std(lstm_inputs[:, :, i])
            if my_std <= 0.00001:
                my_std = 1
            std_factors.append(my_mean)
            std_factors.append(my_std)
        for i in range(args.output_size):
            my_mean = np.mean(target_outputs[:, :, i])
            my_std = np.std(target_outputs[:, :, i])
            if my_std <= 0.00001:
                my_std = 1
            std_factors.append(my_mean)
            std_factors.append(my_std)

    # peform standardization on inputs and targets
    for i in range(args.input_size):
        lstm_inputs[:, :, i] = (
            lstm_inputs[:, :, i] - std_factors[2 * i]
        ) / std_factors[2 * i + 1]
    for i in range(args.output_size):
        target_outputs[:, :, i] = (
            target_outputs[:, :, i] - std_factors[2 * args.input_size + 2 * i]
        ) / std_factors[2 * args.input_size + 2 * i + 1]
    if flag:
        return lstm_inputs, target_outputs, std_factors, sc_inputs
    else:
        return lstm_inputs, target_outputs, sc_inputs
