# SIParCS_ML

## Keras versions
### ML_NN_algorithms.py:

all parameters can be tuned by changed the corresponding parameter to a list (of either randomly generated or hardcoded values)
  values can either be set globally (default) or individually for each net inside of the method call.

every network model will output accuracy, loss, and ROC graphs as well as a .txt file containing a model summary and all of the parameters used in that run, and separate .txt files for the ROC values and thresholds. all generated files will have the same prefix:
  - yymmdd_leadTime_netType
  - text file parameters will have the above format with .txt as the extension
  - all graphs are currently exported as .png files with the above prefix plus either 'accuracy', 'loss', or 'roc' at the end before the extension
  - for models run with the grid search all of the above file names will have an additional number added before the network type (ex. yymmdd_leadTime_X_netType, where X is the iteration of the grid search) so that graphs and parameters for each grid search run can be easily found. for the grid search, each model with a given parameters will be created for t number of trials to combat having a poor model due to the random weight initialization. the files produced by each trial will be yymmdd_leadTime_X.t_netType
  - the grid search loop will also create a file containing the best AUROC value for the Dev set and corresponding searched parameters and search iteration (see previous bullet point) for the model that was run. the best trial number will also be saved in this file for easy finding.

so all files corresponding to a particular run/model
can be easily found together.

## TensorFlow versions
### Training options
Backpropagation.

Evolutionary Algorithm.
