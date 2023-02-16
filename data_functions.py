import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#----------------- Get SSA for from LSTM Results -----------------#
def get_LSTM_SSA(file):
    """Returns SSA of a single experiement from an experiment set
    1 = Zcg
    2 = Roll
    3 = Pitch"""

    # skip rows prevents to remove header
    df = pd.read_csv(
        file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        names=["Time", "Zcg", "Roll", "Pitch"],
        index_col=0,
    )
    return 2 * np.sqrt(df.var())

#----------------- Get SSA from LAMP Results -----------------#

def get_LAMP_SSA(file):
    """Returns SSA of a single experiement from an experiment set
    1 = Zcg
    2 = Roll
    3 = Pitch"""

    df = pd.read_csv(
        file,
        delim_whitespace=True,
        header=2,
        skipfooter=18,
        index_col="Time",
        usecols=["Time", "Zcg", "Rot_X", "Rot_Y"],
        engine="python",
    )
    df = df.rename(columns={"Rot_X": "Roll", "Rot_Y": "Pitch"})
    return 2 * np.sqrt(df.var())


def get_SC_SSA(file):
    """Returns SSA of a single experiement from an experiment set
    1 = Zcg
    2 = Roll
    3 = Pitch"""

    df = pd.read_csv(
        file,
        delim_whitespace=True,
        header=0,
        usecols=["Time", "Zcg", "Roll", "Pitch"],
        index_col="Time",
        skipfooter=18,
        engine="python",
    )
    df = df.drop(labels="sec", axis=0)
    df = df.astype({"Zcg": "float", "Roll": "float", "Pitch": "float"})
    return 2 * np.sqrt(df.var())


#----------------- Create Dataframe for experiement to be turned into a heatmap -----------------#

def LSTMvLAMP_df(experiment="",LSTM_files='', realization=''):
    
    from pathlib import Path
    import pandas as pd

    """Makes a dataframe comparing LSTM and LAMP outputs
        experiment = folder name of the experiment"""

    # Establish paths
    exp_PATH = f'H:\\OneDrive - Massachusetts Institute of Technology\\Thesis\\SimpleCode_to_LAMP_LSTM\\Experiment_ARCHIVE\\Bimodal Test Sets\\{experiment}\\'

    # network options
    # nwOptions = ('original', 'new')
    
    # No experiment or network given
    if experiment == "" or LSTM_files == '':
        return "INVALID EXPERIMENT OR NETWORK"

    # Parse out the variables observed
    v1 = experiment.partition("v")[0]  # Pulls first variable by taking string before "_", spliting at the "v"
    v2 = experiment.partition("v")[2]  # 2nd variable

    # LAMP path
    LAMP_path = exp_PATH + "\\LAMP_files\\"

    # LSTM data
    exp_PATH += LSTM_files + '\\'

    # Set experiment path
    # if network.lower() == "original":
    #     exp_PATH += "\\output_files_MED\\"
    # elif network.lower() == "new":
    #     exp_PATH += "\\output_files_MED2_30_50\\"

    # Make the dataframe for LSTM
    df_LSTM = pd.DataFrame(
        columns=[v1, v2, "Zcg", "Roll", "Pitch"]
    )  # Create cols for the altered variables in experiement and heave, roll, pitch

    # step through every LSTM file in the experiement to get the SSA, and add to dataframe
    for path in Path(exp_PATH).glob(f"*{realization}.txt"):
        # Get the SSA's
        SSA = get_LSTM_SSA(path)
        # Get the variable combo for the file
        trial = path.name.lstrip("lstm_output_for_SC_")
        idx = trial.partition(v1)[2]
        t1 = idx.partition("_")[0]
        if v2 == "s":
            idx = trial.partition(v2)[2]
            t2 = idx.partition("-")[0]
        else:
            idx = trial.partition(v2)[2]
            t2 = idx.partition("_")[0]

        SSA[v1] = float(t1)
        SSA[v2] = float(t2)

        df_LSTM.loc[len(df_LSTM)] = SSA

    df_LSTM = df_LSTM.sort_values(by=[v1, v2])

    # Make the dataframe for LAMP
    df_LAMP = pd.DataFrame(columns=[v1, v2, "Zcg", "Roll", "Pitch"])

    # step through every LAMP file in the experiement to get the SSA, and add to dataframe
    for path in Path(LAMP_path).glob(f"*{realization}.mot"):
        # Get the SSA's
        SSA = get_LAMP_SSA(path)
        # Get the variable combo for the file
        trial = path.name.lstrip("LAMP_")
        idx = trial.partition(v1)[2]
        t1 = idx.partition("_")[0]
        if v2 == "s":
            idx = trial.partition(v2)[2]
            t2 = idx.partition("-")[0]
        else:
            idx = trial.partition(v2)[2]
            t2 = idx.partition("_")[0]

        SSA[v1] = float(t1)
        SSA[v2] = float(t2)

        df_LAMP.loc[len(df_LAMP)] = SSA

    df_LAMP = df_LAMP.sort_values(by=[v1, v2])

    # Establish new, final DF for heatmap
    df_final = pd.DataFrame(columns=[v1, v2, "Zcg Error", "Roll Error", "Pitch Error"])
    df_final[v1] = df_LSTM[v1]
    df_final[v2] = df_LSTM[v2]

    # find the absolute difference between each in the series
    df_final["Zcg Error"] = np.abs(df_LSTM["Zcg"] - df_LAMP["Zcg"])
    df_final["Roll Error"] = np.abs(df_LSTM["Roll"] - df_LAMP["Roll"])
    df_final["Pitch Error"] = np.abs(df_LSTM["Pitch"] - df_LAMP["Pitch"])

    # Return DF, v1 and v2 variables to pivot on, unique lists of v1, v2 to mark training
    return df_final, v1, v2, df_final[v1].unique(), df_final[v2].unique()

#----------------- Creates a Heatmap from a standard dataframe using the experiment listed -----------------#
def makeHeatmap(top_network,bottom_network, experiment, parameter, realization, save_folder, train_files):
    import seaborn as sns
    from matplotlib.patches import Rectangle
    """
    Creates a heatmap of SSA error between unimodal LSTM and same LSTM trained with bimodal seas. Given a df of the form:
    v1      v2    Zcg Error       Roll Error    Pitch Error
    experiment = case file
    network = network trained on
    parameter = parameter we want to see
    realizatino = realization you want to focus on
    save_folder = folder to save in
    train_files = dictionary of training files for each experiment used
    """

    # Renaming Based on Experiment ID
    EXP_ID = {'aa': 'Secondary Heading', 'a': 'Primary Heading', 'hh': 'Secondary Height', 'h': 'Primary Height', 'pp': 'Secondary Period', 'p': 'Primary Period', 's': 'Ship Speed'}
    EXP_ID_units = {'aa': r'$Secondary Heading (^{\circ})$', 'a': r'$Primary Heading (^{\circ})$', 'hh': 'Secondary Height (m)', 'h': 'Primary Height (m)', 'pp': 'Secondary Period (sec)', 'p': 'Primary Period (sec)', 's': 'Ship Speed (knots)'}
    plot_ID = experiment.split('v')

    # Set save path for the figure
    SAVE_PATH = f'H:\\OneDrive - Massachusetts Institute of Technology\\Thesis\\SimpleCode_to_LAMP_LSTM\\Experiment_ARCHIVE\\Transfer Learning Figures\\{save_folder}'

    # Set units for color bar
    if parameter == "Zcg Error":
        units = "meters"
    else:
        units = r"$^{\circ}$"

    # Set the fig and ax
    fig, (axo, axn, axd) = plt.subplots(3, figsize=(16, 16), sharey=True)
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    cbar_ax = fig.add_axes([0.91, 0.425, 0.03, 0.4])
    rbar_ax = fig.add_axes([0.91, 0.125, 0.03, 0.2])
    fig.suptitle(f"{EXP_ID[plot_ID[0]]} vs {EXP_ID[plot_ID[1]]} Absolute SSA {parameter}", fontsize=18)
    fig.supxlabel(f'{EXP_ID_units[plot_ID[0]]}')
    fig.supylabel(f'{EXP_ID_units[plot_ID[1]]}')
    colormap= sns.diverging_palette(0,145,sep=15,as_cmap=True)
    cbar_ax.set_ylabel(units)

    # get the Original dataset
    dfo, v1, v2, v1_vals, v2_vals = LSTMvLAMP_df(experiment, 'output_files_'+top_network, realization)
    resulto = dfo.pivot(index=v2, columns=v1, values=parameter)

    # Set colorbar max value
    vmax = resulto.max().max()

    # get the new dataset
    dfn, v1, v2, _, _ = LSTMvLAMP_df(experiment, 'output_files_'+bottom_network, realization)
    resultn = dfn.pivot(index=v2, columns=v1, values=parameter)

    # Set colorbar max value to that of new if it's more than med
    if resultn.max().max() > vmax:
        vmax = resultn.max().max()

    # Plot original dataset
    g1 = sns.heatmap(
        resulto,cmap="Blues",ax=axo, cbar=True,vmax=vmax,cbar_ax=cbar_ax, # annot = True, fmt = ".2f"
    )
    g1.invert_yaxis()
    if 'BiMod' in top_network:
        g1.set_title("Bimodal Training, N=" + top_network[-1])
    else:
        g1.set_title("Transfer Learning, N=" + top_network[-1])
    g1.set_xlabel('')
    g1.set_ylabel('')

    # get the new dataset
    g2 = sns.heatmap(
        resultn, cmap="Blues", ax=axn, cbar_kws={"label": units}, cbar=True, vmax=vmax, cbar_ax=cbar_ax, # annot = True, fmt = ".2f"
    )
    g2.invert_yaxis()
    if 'BiMod' in bottom_network:
        g2.set_title("Bimodal Training, N=" + bottom_network[-1])
    else:
        g2.set_title("Transfer Learning, N=" + bottom_network[-1])
    g2.set_xlabel('')
    g2.set_ylabel('')

    # Plot a heatmap comparing the difference between the two.  Bigger is better b/c delta_orig - delta_new means new performed much better
    dfc = dfn.copy()
    cols =  dfo.columns.difference([v1,v2])
    dfc[cols] = dfo[cols] - dfn[cols]
    resultc = dfc.pivot(index=v2, columns=v1, values=parameter)

    # set colorbar max and min values
    vmax = resultc.max().max()
    vmin = resultc.min().min()

    g3 = sns.heatmap(
        resultc, cmap=colormap, ax=axd, cbar_kws={'label': units, 'ticks':[vmin,0,vmax], 'format': '%.2f'}, cbar=True, center=0, cbar_ax=rbar_ax, # annot = True, fmt = ".2f"
    )
    g3.invert_yaxis()
    g3.set_title("Delta")
    g3.set_xlabel('')
    g3.set_ylabel('')

    # Find training boxes for later plotting
    # z1 and z2 provide list to map indeces to v1 and v2 vals
    z1 = list(range(0,len(v1_vals)))
    z2 = list(range(0,len(v2_vals)))

    # dictionary index mapping values to heatmap index
    idx1 = dict(zip(v1_vals,z1))
    idx2 = dict(zip(v2_vals,z2))

    # list of tubles for the chosen training records
    hatches = []

    for file in train_files[experiment]:
        val1 = float(file.partition(plot_ID[0])[2].partition('_')[0])
        if plot_ID[1] == 's':
            val2 = float(file.partition(plot_ID[1])[2].partition('-')[0])
        else:
            val2 = float(file.partition(plot_ID[1])[2].partition('_')[0])

        # Plot the rectangles of training records on each heatmap
        axo.add_patch(Rectangle((idx1[val1],idx2[val2]),1,1, fill=False, edgecolor='orange', lw=3))
        axn.add_patch(Rectangle((idx1[val1],idx2[val2]),1,1, fill=False, edgecolor='orange', lw=3))
        axd.add_patch(Rectangle((idx1[val1],idx2[val2]),1,1, fill=False, edgecolor='orange', lw=3))
        hatches.append((idx1[val1],idx2[val2]))

    # Save the figure
    plt.savefig(f"{SAVE_PATH}\\{experiment}_{save_folder}_{parameter}_{realization}.png")
