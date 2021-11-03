# SimpleCode_to_LAMP_LSTM
Train and run an LSTM in python which creates a map from SimpleCode output to LAMP output.

You will need three types of files for each realization: A SimpleCode .mot file, a LAMP .mot file,
and LAMP's .wav file. The SimpleCode .mot files should all be in the same folder, and the LAMP .mot
and .wav files should have their own folder.

The "input_template.py" file can be named anything. You can have multiple copies of this file
with different names for different settings. Once you have the inputs you want, go into main.py
and alter the following line (near the top, around line 16):

from **input_template** import UserInputArgs, PlottingArgs, DataInfoArgs, DerivedArgs

Replace the bolded **input_template** with the name of your desired input file, excluding the ".py".

In the input_template file, there are 4 classes, three of which can/should be modified. Leave the 
DerivedArgs class alone.

UserInputArgs has the parameters that pertain to the neuralnetwork architecture and whether or not
you will train a new network or load in a previously saved network.

PlottingArgs is used to specify which realizations from the training, validation, and testing set
you want plotted, as well as a few plot style preferences.

DataInfoArgs is where you will specify all of the SimpleCode and LAMP file names. There are six attributes,
self.train_lamp, .train_sc, .val_lamp, .val_sc, .test_lamp, .test_sc. The user must construct each of these
to be a list of filepath/filenames, excluding any ".mot" or ".wav" (those are accounted for later). This
can be done manually, or with some type of a loop.

(Currently, all six of these attributes must have at least one filepath/filename specified, otherwise it throws
an error. I'll fix this later).

A quick note on the difference between validation and testing data: When training a neural network, the objective
is to train it such that its behavior is generalizable to data outside of its training set. The validation data
helps us do this in two ways. First, it helps us determine hyper parameters. Say I have 10 combinations of hyper
parameters. I will train a separate neural network for each one, and then see how well those 10 neural networks
perform on the validation data. Whichever performs the best gives me an indication of which hyper parameters I
should use. Secondly, the validation data helps us to know when to stop training a neural network during a 
particular session. In some cases, we can keep on training a neural network indefinitely and the error on the 
training data will approach 0. However, this is often the result of the neural network effectively memorizing
the training data without learning the _true_ generalizable trends. This is called "overfitting". To avoid
an overfit neural network, we evaluate the performance of the neural network on the validation data, which it has
never trained on. When the error on the validation data stops going down for enough training epochs, we cease training,
and claim the version of the neural network that did best of the validation data.

The test data is reserved separately from the validation data sort of as another layer guarding against overfitting.
When we have finished figuring out what hyper parameters we want, we've done the amount of training that we think
is best based on the validation data, and we would like to report the accuracy or performance of the network, we then
use the test data. It's a way of better judging the generalizability of the model, since it hasn't ever seen the
test data before, either in training or in hyper parameter tuning.
