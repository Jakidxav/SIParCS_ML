"""
Authors: Negin Sobhani, Jakidxav, Karen Stengel
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

#will allow for files to
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#path for data output. each file should contain all used params after training and metrics
outputDir =  "./data/Conv/weights/"

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
    train_data = None
    train_label = None
    dev_data = None
    dev_label = None
    test_data = None
    test_label = None
    outputFile = ""
    bestFile = ""

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d_")
    #print(date)

    #for all dataset directories & all data in each: need to walk through the various dierctories that contain each dataset
    #can change this directory once run location on cheyenne is selected

    outputDL = date + "_30_"
    outputFile = outputDir + outputDL

    X_train_filename = '/glade/work/jakidxav/IPython/X/30_lead/X_train/X_train.txt'
    X_dev_filename = '/glade/work/jakidxav/IPython/X/30_lead/X_dev/X_dev.txt'
    X_val_filename = '/glade/work/jakidxav/IPython/X/30_lead/X_val/X_val.txt'

    Y_train_filename = '/glade/work/jakidxav/IPython/Y/Y_train/station0/Y_train.txt'
    Y_dev_filename = '/glade/work/jakidxav/IPython/Y/Y_dev/station0/Y_dev.txt'
    Y_val_filename = '/glade/work/jakidxav/IPython/Y/Y_val/station0/Y_val.txt'

    with open(X_train_filename, 'rb') as f:
        train_data = pickle.load(f)

    with open(X_dev_filename, 'rb') as g:
        dev_data = pickle.load(g)

    with open(X_val_filename, 'rb') as h:
        test_data = pickle.load(h)

    with open(Y_train_filename, 'rb') as i:
        train_label = pickle.load(i)

    with open(Y_dev_filename, 'rb') as j:
        dev_label = pickle.load(j)

    with open(Y_val_filename, 'rb') as k:
        test_label = pickle.load(k)

    #reshape all data files.
    train_data2 = train_data.reshape(-1,120,340, 1)
    dev_data2 = dev_data.reshape(-1,120,340,1)
    test_data2 = test_data.reshape(-1,120,340,1)

    #set up file name for best parameters file. willl have to change things to write to file as the
    #search parameters are chosen/optimized and others are being tried.
    bestFile = outputFile + "best.txt"
    bfile = open(bestFile, "w+")
    #bfile.write("epochs tried: " + " ".join(str(x) for x in epochs) + "\n")
    #bfile.write("dropouts tried: " + " ".join(str(x) for x in dropout) + "\n")
    #bfile.write("learning rates tried: " + " ".join(str(x) for x in learningRate) + "\n")

    #best scores for each net and the associated parameters
    #will also have to change the param lists depending on which params are being optimized
    #bestDnnAUROC = 0
    #bestDnnParams = [epochs[0], learningRate[0]]
    #bestDnnSearchNum = 0

    bestCnnAUROC = 0
    #bestCnnParams = [epochs[0], learningRate[0]]
    bestCnnSearchNum = 0

    #bestRnnAUROC = 0
    #bestRnnParams = [epochs[0], learningRate[0]]
    #bestRnnSearchNum = 0

    #bestAlexAUROC = 0
    #bestAlexParams = [epochs[0], learningRate[0]]
    #bestAlexSearchNum = 0


    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    start = time.time()
    i = 0
    for w in posWeight:

        outputSearch = outputFile + str(i) + "_"
        #denseNN, dnnAUROC = dnn([16,16], dropout, learningRate, momentum, decay, boolNest, boolAdam, beta_1, beta_2, epsilon, amsgrad,219,train_data2, train_label,dev_data2, dev_label, outputSearch, i, batch) #these are all negins values right now.
        model = None
        modelAUROC = 0
        bestTry = 0
        for t in range(trials):

            recurrNN, rnnAUROC = rnn([20,60],kernel, pool, strideC, strideP, dropout, 0.123, momentum, 1.0e-4, boolNest,True, boolAdam,beta_1, beta_2, epsilon, amsgrad, 17, train_data2, train_label, dev_data2, dev_label,outputSearch, i, batch,w)
            if rnnAUROC > modelAUROC:
                model = recurrNN
                modelAUROC = rnnAUROC
                bestTry = t
        model.save(outputSearch + str(bestTry)+'.h5')

        if modelAUROC > bestRnnAUROC:
            bestRnnAUROC = modelAUROC
            bestRnnParams = [w]
            bestRnnSearchNum = i
        i += 1

    #bfile.write("best DNN AUROC for dev set: " + str(bestDnnAUROC) + "\n")
    #bfile.write("best DNN search iteration for dev set: " + str(bestDnnSearchNum) + "\n")
    #bfile.write("best parameters for DNN: " + " ".join(str(x) for x in bestDnnParams) + "\n\n")

    bfile.write("best CNN AUROC for dev set: " + str(bestCnnAUROC) + "\n")
    bfile.write("best CNN search iteration for dev set: " + str(bestCnnSearchNum) + "\n")
    #bfile.write("best parameters for CNN: " + " ".join(str(x) for x in bestCnnParams) + "\n\n")

    #bfile.write("best RNN AUROC for dev set: " + str(bestRnnAUROC) + "\n")
    #bfile.write("best RNN search iteration for dev set: " + str(bestRnnSearchNum) + "\n")
    #bfile.write("best parameters for RNN: " + " ".join(str(x) for x in bestRnnParams) + "\n\n")

    #bfile.write("best Alex AUROC for dev set: " + str(bestAlexAUROC) + "\n")
    #bfile.write("best Alex search iteration for dev set: " + str(bestAlexSearchNum) + "\n")
    #bfile.write("best parameters for Alex: " + " ".join(str(x) for x in bestAlexParams) + "\n\n")

    print("runtime ",time.time() - start, " seconds")
    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
