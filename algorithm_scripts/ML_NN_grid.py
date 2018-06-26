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


#for numerical values use a random distribution. format: sp_randint(min, max)
batch_size = sp_randint(50, 256)
dnnNeuronLayer = [16,16]
cnnNeuronLayer = [6,16,120,84]

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

    for layer in range(len(dnnNeuronLayer)):
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dropout(drop))
        denseModel.add(Dense(dnnNeuronLayer[layer], kernel_regularizer=l2(0.0001), activation = 'relu'))

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

def cnn(optimizer = 'adam'):
    '''
    implements a convolutional neural network and creates files with parameters and plots

    Arguments:

    Returns:
        convModel : a trained keras convolutional network

    Example:
        cnn([32,64, 1000, 1], [5,5], [2,2], [1,1], [1,1], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
        lenet would be: cnn([6,16,120,84], [5,5], [2,2], [1,1], [2,2], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
        alexnet: https://gist.github.com/JBed/c2fb3ce8ed299f197eff
    '''
    print("convoultional neural network")

    #make sure all lists are the same length s.t. the for loops for setting up dont break
    assert (len(kernel) == len(strideC))
    assert (len(pool) == len(strideP))

    #initilaize model with Sequential
    convModel = Sequential()

    #add first conv and pooling layers
    convModel.add(Conv2D(cnnNeuronLayer[0], kernel_size=(kernel[0], kernel[0]), strides = strideC[0],padding = 'same',activation='relu', input_shape=train_data[0].shape))
    convModel.add(MaxPooling2D(pool_size=(pool[0], pool[0]), strides=(strideP[0], strideP[0]),padding = "valid"))

    for layer in range(1, len(cnnNeuronLayer) - 2):
        print(layer, strideP[layer])

        convModel.add(Conv2D(cnnNeuronLayer[layer], kernel_size = (kernel[layer],kernel[layer]), strides = strideC[layer], padding = 'same', activation='relu'))
        convModel.add(MaxPooling2D(pool_size=(pool[layer], pool[layer]), strides=(strideP[layer], strideP[layer]), padding = "valid"))

    convModel.add(Conv2D(cnnNeuronLayer[len(cnnNeuronLayer) - 2], kernel_size = kernel[len(kernel) - 1], strides = strideC[len(strideC) - 1], activation = 'relu'))
    convModel.add(Dropout(drop))
    convModel.add(Flatten())
    convModel.add(Dense(cnnNeuronLayer[len(cnnNeuronLayer) - 1], kernel_regularizer=l2(0.0001), activation='relu'))
    convModel.add(Dropout(drop))
    convModel.add(Dense(1, kernel_regularizer=l2(0.0001), activation='sigmoid'))
    #save model to a file
    with redirect_stdout(file):
        convModel.summary()

    #define optimizer and compile
    '''
    if boolAdam:
        opt_conv = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    else:
        opt_conv = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)
    '''
    convModel.compile(loss=binary_crossentropy,optimizer=opt_conv,metrics=[binary_accuracy])
    '''
    #fit model
    conv_hist = convModel.fit(train_data, train_label,batch_size=256,epochs=iterations,verbose=1,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = convModel.predict(train_data).ravel()
    dev_pred = convModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(conv_hist, outputFile, "Conv Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)
    '''
    return convModel

#main stuff
    #this should read in each dataset and call the NN algorithms.
if __name__ == "__main__":
    print("you are in main.")

    # below methods use https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    #as a guide line. using their printing method for best params.

    #dnn random grid search
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


    #cnn random grid search
    cmodel = KerasClassifier(build_fn=cnn, verbose=1)

    #rnn random grid search
    #rmodel = KerasClassifier(build_fn=rnn, verbose=1)
