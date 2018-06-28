# SIParCS_ML

## Use of ML_NN_algorithms.py:
ML_NN_algorithms.py takes in 2 Arguments - netType and stationNum
  - netType can be 'dense', 'conv', 'recur', 'alex', or 'all' for running
  the dense net, conv net, recurrent net, alexNet, or all of them during the grid search, respectively.

  - stationNum should take the form of 'stationX' where X is an integer value. this allows for a selection of
  which station is selected as labels regardless of lead time.

all lead times will be run automatically if they are in the same directory as ML_NN_algorithms.py
  this can changed directly in the script in __main__ or by only having one lead time folder in the directory.
  all lead time files should be in the form 'X_lead' where X an integer representing how many days are used in the lead time.

all parameters can be tuned by changed the corresponding parameter to a list (of either randomly generated or hardcoded values)
  values can either be set globally (default) or individually for each net inside of the method call.

every network model will output accuracy, loss, and ROC graphs as well as a .txt file containing a model summary and all of the
parameters used in that run. all generated files will have the same prefix:
  - yymmdd-hhmm_leadTime_netType
  - text file parameters will have the above format with .txt as the extension
  - all graphs are currently exported as .png files with the above prefix plus either 'accuracy', 'loss', or 'roc' at the end before the extension
  - for models run with the grid search all of the above file names will have an additional number added before the network type (ex. yymmdd-hhmm_leadTime_X_netType, where X is the iteration of the grid search) so that graphs and parameters for each grid search run can be easily found
  - the grid search loop will also create a file containing the best AUROC value for the Dev set and corresponding searched parameters and search iteration (see previous bullet point) for the model that was run.

so all files corresponding to a particular run/model
can be easily found together.
