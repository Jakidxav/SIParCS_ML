"""
Authors: Negin Sobhani, Jakidxav, Karen Stengel

This is the main training script for our repository. We start by setting all of our hyperparameters that we got through a hyperparameter search.
Change the lead time from 20, 30, 40, or 50 days in advance, and the correct data will be loaded and the program will redirect output to the proper folder.
We then specify a string representing the type of architecture that we want, 
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

#constants declaration
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)
posWeight = [2,4,8,16]
trials = 3
#hyperparameters and paramters
#SGD parameters
dropout = 0.5
momentum = 0.99

learningRate = 0.011
epochs = 233
decay = 1.0e-4
batch = 128

boolNest = True

#parameters for conv/pooling layers
strideC = [5,5, 1]
strideP = [2,2]
kernel = [5, 5,1]
pool = [2,2]

#parameters for Adam optimizaiton
boolAdam = True #change to false if SGD is desired

beta_1= 0.9
beta_2= 0.999

epsilon=None
amsgrad=False




#main method: this will read in each dataset and call the NN algorithms.
if __name__ == "__main__":
    print("you are in main.")

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d_")

    #for all dataset directories & all data in each: need to walk through the various dierctories that contain each dataset
    #can change this directory once run location on cheyenne is selected

    #specificy which type of architecture you are using
    #choices are: conv, recur, dense, alex
    model_ = 'conv'

    #path for data output. each file should contain all used params after training and metrics
    outputDir =  "./data/{}/weights/".format(model_)

    #specify lead time and date that you ran the model
    lead_time = '30'
    outputDL = date + "_{}_".format(lead_time)
    outputFile = outputDir + outputDL

    #data directories on disk
    X_train_filename = '/glade/work/jakidxav/IPython/X/{}_lead/X_train/X_train.txt'.format(lead_time)
    X_dev_filename = '/glade/work/jakidxav/IPython/X/{}_lead/X_dev/X_dev.txt'.format(lead_time)
    X_val_filename = '/glade/work/jakidxav/IPython/X/{}_lead/X_val/X_val.txt'.format(lead_time)

    Y_train_filename = '/glade/work/jakidxav/IPython/Y/Y_train/station0/Y_train.txt'
    Y_dev_filename = '/glade/work/jakidxav/IPython/Y/Y_dev/station0/Y_dev.txt'
    Y_val_filename = '/glade/work/jakidxav/IPython/Y/Y_val/station0/Y_val.txt'

    #open all data
    train_data, train_label = load_data(X_train_filename, Y_train_filename)
    dev_data, dev_label = load_data(X_dev_filename, Y_dev_filename)
    test_data, test_label = load_data(X_val_filename, Y_val_filename)

    #reshape all data files.
    train_data2 = train_data.reshape(-1,120,340, 1)
    dev_data2 = dev_data.reshape(-1,120,340,1)
    test_data2 = test_data.reshape(-1,120,340,1)

    #set up file name for best parameters file. willl have to change things to write to file as the
    #search parameters are chosen/optimized and others are being tried.
    bestFile = outputFile + "best.txt"
    bfile = open(bestFile, "w+")
    bfile.write("epochs tried: " + " ".join(str(x) for x in epochs) + "\n")
    bfile.write("dropouts tried: " + " ".join(str(x) for x in dropout) + "\n")
    bfile.write("learning rates tried: " + " ".join(str(x) for x in learningRate) + "\n")

    #best scores for each net and the associated parameters
    #will also have to change the param lists depending on which params are being optimized
    bestAUROC = 0
    bestParams = [epochs[0], learningRate[0]]
    bestSearchNum = 0

    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    start = time.time()
    i = 0

    #loop over the class weights; we are trying to see if weighting the sparse hot days 
    #in our data set improves prediction skil
    for w in posWeight:

        outputSearch = outputFile + str(i) + "_"
        
        bestModel = None
        AUROC = 0
        bestTry = 0

        #now loop over trials; we want to initialize t different networks of a given architecture, and take then save the best one
        for t in range(trials):

	    #create dense neural network
	    if model_ == 'dense':
	        model, modelAUROC = dnn([16,16], dropout, learningRate, momentum, decay, boolNest, boolAdam, beta_1, beta_2, epsilon, amsgrad, 219, train_data2, train_label, dev_data2, dev_label, outputSearch, i, batch) 

	    #create LeNet5 architecture
	    elif model_ == 'conv':
	        model, modelAUROC = cnn([20, 60], kernel, pool, strideC, strideP, dropout, learningRate, momentum, decay, boolNest, boolAdam, beta_1, beta_2, epsilon, amsgrad, epochs, train_data2, train_label, dev_data2, dev_label, outputSearch, i, batch, w)

	     #create lstm architecture
	    elif model_ == 'recur':
	        model, modelAUROC = rnn([20,60], kernel, pool, strideC, strideP, dropout, learningRate, momentum, decay, boolNest, True, boolAdam, beta_1, beta_2, epsilon, amsgrad, epochs, train_data2, train_label, dev_data2, dev_label, outputSearch, i, batch, w)
     
	    #create AlexNet architecture
	    else:
	        model, modelAUROC = alex([20,60], kernel, pool, strideC, strideP, dropout, learningRate, momentum, decay, boolNest, boolAdam, beta_1, beta_2, epsilon, amsgrad, epochs, train_data2, train_label, dev_data2, dev_label,outputSearch, i, batch, w)

            #compare ROC score to saved ROC score; the first model's ROC will always be saved since we have initilized modelAUROC to 0
            if modelAUROC > AUROC:
                #if the ROC score is better, save the model, update the best ROC score, and the best trial number
                bestModel = model
                AUROC = modelAUROC
                bestTry = t
        
        #save the best model per t trials
        model.save(outputSearch + str(bestTry)+'.h5')

        #now compare the best model per trials to those of different class weights
        if modelAUROC > bestAUROC:
            #once again: if the ROC score is better, save the model, update the best ROC score, and the best class weights number
            bestAUROC = modelAUROC
            bestParams = [w]
            bestSearchNum = i

        #update class weights iterator
        i += 1
        
        bfile.write("best {} AUROC for dev set: ".format(model_) + str(bestAUROC) + "\n")
        bfile.write("best {} search iteration for dev set: ".format(model_) + str(bestSearchNum) + "\n")
        bfile.write("best parameters for {}: ".format(model_) + " ".join(str(x) for x in bestParams) + "\n\n")

    print("runtime ",time.time() - start, " seconds")
