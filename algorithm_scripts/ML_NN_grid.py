'''
this script uses the same methods as ML_NN_algorithms.py but uses
scikit learn GridSearchCV to find best parameters. should be used to narrow down hyperparameters
for use in the ML_NN_algorithms.py
'''

"""

@author Negin Sobhani, Joshua Driscol, Karen Stengel

this script is designed to use keras with Tensorflow as the backend.
it should be used in conjunction with data_processing.py which should set up/create necessary
datasets from the McKinnon et al. dataself plus any other data added to the set.

this script will take the premade datasets and run them through the various ML NN algorithms
and return ROC scores; can use the binary_accuaracy method in keras as a starting pointself.
This script will also include optimizers so that we can get the best
scores possible.

algorithms to include:
    dense nn - already created by Negin so will just need to try optimizing
    CNN - also already tried by Negin so will need to try optimizing.
    RNN - use LSTM. look into if the data needs to be reshaped
    siamese nn - possibly use? need to look into best implementation...
    RBFN - ? maybe use? could allow for an interesing analysis....

    will also need to save models once they are fully trained/work

datasets:
    should be set up with various lead prediction times. ex: 10, 30, 45, 60 days in advance
    should also use a control set with randomly shuffled labels

try with other stations; could do individually or as a multiclassification with ~20

try with soil temperature; need separate NN and merge

figure out how file output works with grid search
alexnet; with loaded weights from alexnet website

find time and space complexity - ask alessandro
"""
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from keras.models import Sequential
import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, TimeDistributed, LSTM
from keras.layers import Dropout, BatchNormalization
from keras.metrics import binary_accuracy
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
import sklearn.metrics as skm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from tensorflow import metrics as tf
import pickle
import os
import datetime

train_data = None
train_label = None
dev_data = None
dev_label = None
test_data = None
test_label = None
outputFile = ""

for folder in os.listdir('.'):

    #extract the data from all the necessary files for the given lead time
    # need to change
    if (not '.' in folder) and '_' in folder:

        for f in os.listdir(folder):

            if not f.startswith('.'):

                #_norm
                with open(folder + "/" + f + "/" + f + ".txt", 'rb') as file:
                    if f == 'X_train':
                        train_data = pickle.load(file)
                    if f == 'X_dev':
                        dev_data = pickle.load(file)
                    if f == 'X_val':
                        test_data = pickle.load(file)
                    if f == 'Y_train':
                        train_label = pickle.load(file)
                    if f == 'Y_dev':
                        dev_label = pickle.load(file)
                    if f == 'Y_val':
                        test_label = pickle.load(file)

#reshape all data files.
train_data2 = train_data.reshape(-1,120,340, 1)
dev_data2 = dev_data.reshape(-1,120,340,1)
test_data2 = test_data.reshape(-1,120,340,1)

#hyperparameters and paramters. use as list for GridSearchCV options.
#SGD parameters
drop = 0.5
learningRate = 0.001
momentum = 0.99
decay = 1e-4
boolNest = True

epochs = sp_randint(150,250)

#parameters for conv/pooling layers
strideC = [5,5,1]
strideP = [2,2]
kernel = [5, 5,1]
pool = [2,2]

#parameters for Adam optimizaiton
boolAdam = True #change to false if SGD is desired
beta_1=0.9
beta_2=0.999
epsilon=None
amsgrad=False


#for numerical values use a random distribution 
batch_size = sp_randint(50, 256)
neuronLayer = [16,16]

optimizer = ['SGD', 'adam']

#GridsearchCV dictionary params
param_grid = param_grid = dict(optimizer=optimizer, epochs=epochs, batch_size = batch_size)

#dense nn
    # based off of dnn from Negin. just need to focus on optimizing
def dnn(optimizer = 'adam'):
    '''
    implements a dense neural network. also outputs info to a file.

    Returns:
        denseModel : a trained keras dense network

    Example :
        dnn([16,16], param_grid, train_data, train_label, dev_data, dev_label, outputDL)
    '''
    print("dense neural network")
    #initilaize model with Sequential()
    denseModel = Sequential()
    #add first layers
    denseModel.add(AveragePooling2D(pool_size = (32,32), input_shape = train_data2[0].shape)) # negin used this as the first layer. need to double check syntax
    denseModel.add(Flatten())

    for layer in range(len(neuronLayer)):
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dropout(drop))
        denseModel.add(Dense(neuronLayer[layer], kernel_regularizer=l2(0.0001), activation = 'relu'))

    denseModel.add(Dense(1, kernel_regularizer=l2(0.0001), activation = 'sigmoid'))
    #define optimizer
        #adam optimizer
    #opt_dense = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        #SGD optimizer
    #opt_dense = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)
    #opt_dense = Adam(lr = learnRate)

    denseModel.summary()

    #compile
    denseModel.compile(optimizer, binary_crossentropy, metrics=['accuracy'])
    '''
    dense_hist = denseModel.fit(train_data, train_label, batch_size=256, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = denseModel.predict(train_data).ravel()
    dev_pred = denseModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(dense_hist, outputFile, "Dense Neural Net",fpr_train, tpr_train, fpr_dev, tpr_dev)
    '''
    return denseModel

#main stuff
    #this should read in each dataset and call the NN algorithms.
if __name__ == "__main__":
    print("you are in main.")


    model = KerasClassifier(build_fn=dnn, verbose=1)
    n_iter_search = 20
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter_search)

    grid_result = random_search.fit(train_data2, train_label)
# summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
