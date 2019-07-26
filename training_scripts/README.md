This directory illustrates how we trained our dense, recurrent, and convolutional neural networks on the training and development sets, and then made predictions on the testing set.

A suggested order to look at the files is:
- ML_NN_algorithms: builds a neural network's architecture, and trains on the training and development sets for a given lead time and set of hyperparameters
- keras_loadModel: shows how we can load in a model from a checkpoint and continue training it, as well as make an ROC curve and confusion matrix from that model's predictions
- openModel_makePredictions: gives an example of how to make a prediction on the test set for a given model and lead time
- modelTests: predicts on the testing set for both the LSTM and CNN across all lead times, creates ROC plots
- run_script.sh: runs our main training script while specifying resources like cluster node, GPU usage, and wall clock time

Some helper method files are included as .py files:
- build_models: creates a neural network of a specific architecture given a set of hyperparameters
- plotting: makes an ROC curve, accuracy and loss vs epochs plots, and confusion matrices for a given neural network
- helper_methods: writes model hyperparameters to file so that we can reconstruct a model if we want, and implements a much cleaner way to load in a set of data and labels for either the training, development, or testing sets
