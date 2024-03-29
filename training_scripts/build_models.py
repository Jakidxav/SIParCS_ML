"""
This helper method constructs different neural network architectures based on a set of hyperparameters. It includes methods for dense, recurrent, and convolutional neural networks with implementations of LeNet and AlexNet architecture.
"""


from contextlib import redirect_stdout
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import pickle
import os
import datetime
import time
import random
import sys

from keras.models import Sequential, Model, save_model, load_model
import keras.backend as K
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, TimeDistributed, LSTM, Dropout, BatchNormalization
from keras.metrics import binary_accuracy
from keras.losses import  binary_crossentropy
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
import sklearn.metrics as skm
from tensorflow import metrics as tf
import h5py

from plotting import *
from helper_methods import *
from build_models import *


'''
implements a dense neural network. also outputs info to a file.

Arguments:
neuronLayer : array containing the number of neurons perlayer excluding input layer
drop : % of neurons to drop. must be 0 < & < 1
learnRate : learning rate. should be a float
momentum : momentum. should be float
decay : decay. also float
boolNest : boolean representing if nesterov is used for optimizing
iterations : number of iterations to train the model
train_data : numpy array data to train on
train_label : numpy array labels of training data
test_data : numpy array of test data
test_label : numpy array of test labels
outputDL : path for data output

Returns:
denseModel : a trained keras dense network

Example :
dnn([16,16], 0.5, 0.0001, 0.99, 1e-4, True, train_data, train_label, dev_data, dev_label, outputDL)
'''

def dnn(neuronLayer, drop, learnRate, momentum, decay,boolAdam, boolNest, b1, b2, epsilon, amsgrad,iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum, batch, posWeight):
    print("dense neural network")
    #set final output name/location.
    outputFile = outputDL + "dnn"
    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file, neuronLayer, iterations, None, boolAdam, boolNest, drop, [None], [None], [None], [None], momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight)

    #initilaize model with Sequential()
    denseModel = Sequential()
    #add first layers
    denseModel.add(AveragePooling2D(pool_size = (32,32), input_shape = train_data[0].shape)) # negin used this as the first layer. need to double check syntax
    denseModel.add(Flatten())

    for layer in range(len(neuronLayer)):
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dropout(drop))
        denseModel.add(Dense(neuronLayer[layer], kernel_regularizer=l2(0.0001), activation = 'relu'))

    denseModel.add(Dense(1, kernel_regularizer=l2(0.0001), activation = 'sigmoid'))
    #define optimizer
    if boolAdam:
        #adam optimizer
        opt_dense = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    else:
        #SGD optimizer
        opt_dense = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)
    #opt_dense = Adam(lr = learnRate)
    with redirect_stdout(file):
        denseModel.summary()

    #compile
    denseModel.compile(opt_dense, binary_crossentropy, metrics=[binary_accuracy])

    dense_hist = denseModel.fit(train_data, train_label, batch_size=256, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = denseModel.predict(train_data).ravel()
    dev_pred = denseModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)

    makePlots(dense_hist, outputFile, "Dense Neural Net",fpr_train, tpr_train, fpr_dev, tpr_dev, train_pred, dev_pred)

    denseModel.save(outputFile+ '.h5')

    return denseModel, skm.auc(fpr_dev,tpr_dev)



'''
Implements a convolutional neural network and creates files with parameters and plots

Arguments:
neuronLayer : array containing the number of neurons perlayer excluding input layer
kernel : array of size of conv kernel
pool : array of pool_size amount
strideC : array of lengths of conv stride
strideP : array of lengths of pool stride
drop : % of neurons to drop
learnRate : learning rate
momentum : momentum to use
decay : decay rate to used
boolNest : use nestov or not
iterations : number of iterations to train the model
train_data : data to train on (numpy array)
train_label : labels of training data (numpy array)
outputDL : path for data output

Returns:
convModel : a trained keras convolutional network

Example:
cnn([32,64, 1000, 1], [5,5], [2,2], [1,1], [1,1], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
lenet would be: cnn([6,16,120,84], [5,5], [2,2], [1,1], [2,2], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
'''
def cnn(neuronLayer, kernel, pool,strideC, strideP, drop, learnRate, momentum, decay,boolNest,boolAdam, b1, b2, epsilon, amsgrad,iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum, batch, posWeight):
    print("convoultional neural network")

    #make sure all lists are the same length s.t. the for loops for setting up dont break
    assert (len(kernel) == len(strideC))
    assert (len(pool) == len(strideP))

    outputFile = outputDL + "cnn"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,neuronLayer, iterations, None, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight)

    #initilaize model with Sequential
    convModel = Sequential()

    #add first conv and pooling layers
    convModel.add(Conv2D(neuronLayer[0], kernel_size=(kernel[0], kernel[0]), strides = strideC[0],padding = 'same',activation='relu', input_shape=train_data[0].shape))
    convModel.add(MaxPooling2D(pool_size=(pool[0], pool[0]), strides=(strideP[0], strideP[0]),padding = "valid"))

    for layer in range(1, len(neuronLayer) - 2):

        convModel.add(Conv2D(neuronLayer[layer], kernel_size = (kernel[layer],kernel[layer]), strides = strideC[layer], padding = 'same', activation='relu'))
        convModel.add(MaxPooling2D(pool_size=(pool[layer], pool[layer]), strides=(strideP[layer], strideP[layer]), padding = "valid"))

    convModel.add(Conv2D(neuronLayer[len(neuronLayer) - 2], kernel_size = kernel[len(kernel) - 1], strides = strideC[len(strideC) - 1], activation = 'relu'))
    convModel.add(Dropout(drop))
    convModel.add(Flatten())
    convModel.add(Dense(neuronLayer[len(neuronLayer) - 1], kernel_regularizer=l2(0.0001), activation='relu'))
    convModel.add(Dropout(drop))
    convModel.add(Dense(1, kernel_regularizer=l2(0.0001), activation='sigmoid'))
    #save model to a file
    with redirect_stdout(file):
        convModel.summary()

    #define optimizer and compile
    if boolAdam:
        opt_conv = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    else:
        opt_conv = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)

    convModel.compile(loss=binary_crossentropy,optimizer=opt_conv,metrics=[binary_accuracy])

    #fit model
    conv_hist = convModel.fit(train_data, train_label,batch_size=batch,epochs=iterations,verbose=2,validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = convModel.predict(train_data).ravel()
    dev_pred = convModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    makePlots(conv_hist, outputFile, "Conv Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    convModel.save(outputFile+ '.h5')

    return convModel, skm.auc(fpr_dev,tpr_dev)



'''
Implements a recurrent neural network based on the LeNet architecture: 

Arguments:
neuronLayer : array containing the number of neurons perlayer excluding input layer
kernel : sixe of convolutional kernel
pool : size of pool
strideC : size of conv stride
strideP : size of pooling stride
dropout : % of neurons to dropout. set to 0 if not using dropout
learnRate : learning rate for optimization
momentum : momentum to use
decay : decay rate to used
boolNest : use nestov or not
iterations : number of iterations to train the model
iterations : number of iterations to train the model
boolLSTM : boolean representing to use LSTM net or simple RNN
train_data : data to train on
train_label : labels of training data
dev_data : data for dev set/validation
dev_label : labels for the dev set
outputDL : path for data output

Returns:
recurModel : a trained keras recurrent network
'''
def rnn(neuronLayer, kernel, pool, strideC, strideP, drop, learnRate, momentum, decay,boolNest,boolLSTM, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum, batch, posWeight):
    print("recurrent neural network")
    outputFile = outputDL + "rnn"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight)

    #set up model with sequential
    recurModel = Sequential()
    #create a cnn
    #should allow for the images to be processed similar to movie frames.
    recurModel.add(Conv2D(neuronLayer[0], kernel_size=(kernel[0], kernel[0]), strides = strideC[0],padding = 'same',activation='relu', input_shape=train_data[0].shape))
    recurModel.add(MaxPooling2D(pool_size=(pool[0], pool[0]), strides=(strideP[0], strideP[0]),padding = "valid"))
    recurModel.add(TimeDistributed(Flatten()))

    #setup rnn/lstm
    for layer in range(1,len(neuronLayer)):
        if boolLSTM:
            recurModel.add(LSTM(neuronLayer[layer], return_sequences = True))
        else:
            print("RNN?")
    if drop > 0:
            recurModel.add(Dropout(drop))

    recurModel.add(Flatten())
    recurModel.add(Dense(1, kernel_regularizer=l2(0.0001), activation = 'sigmoid'))

    #save model to a file
    with redirect_stdout(file):
        recurModel.summary()

    #compile and train the model
    if boolAdam:
        opt_rnn = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    else:
        opt_rnn = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)

    recurModel.compile(loss=binary_crossentropy,optimizer=opt_rnn,metrics=[binary_accuracy])
    recur_hist = recurModel.fit(train_data, train_label,batch_size=256,epochs=iterations,verbose=1,validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = recurModel.predict(train_data).ravel()
    dev_pred = recurModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    makePlots(recur_hist, outputFile, "LSTM Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    recurModel.save(outputFile+ '.h5')

    return recurModel, skm.auc(fpr_dev,tpr_dev)




"""
Manual implementation of AlexNet: https://gist.github.com/JBed/c2fb3ce8ed299f197eff
"""
def alex(learnRate, momentum, decay, boolNest, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputSearch, searchNum, batch, posWeight):

    outputFile = outputDL + "alex"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,[4096, 4096, 1000], 1, None, True, False, 0.4, [11, 11, 3,3,3], [2,2,1], [1,1,1,1], [2,2,1], None, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum,posWeight)

    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(train_data[0].shape), kernel_size=(11,11),strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if boolAdam:
        #adam optimizer
        opt_alex = Adam(lr=learnRate, beta_1=b1, beta_2=b2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    else:
        #SGD optimizer
        opt_alex = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)

    with redirect_stdout(file):
        model.summary()

    # (4) Compile
    model.compile(loss='binary_crossentropy', optimizer=opt_alex, metrics=[binary_accuracy])

    # (5) Train
    alex_hist = model.fit(train_data, train_label, batch_size=64, epochs=iterations, verbose=2, validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = model.predict(train_data).ravel()
    dev_pred = model.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    makePlots(alex_hist, outputFile, "Alex Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    model.save(outputFile+ '.h5')

    return model, skm.roc_curve(dev_label, dev_pred)

