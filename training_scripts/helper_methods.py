from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
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


"""
This method writes the hyperparameters used by a model to file. This is beneficial in case you want to re-create a model that is performing well.

We use this method in multiple scripts and Jupyter Notebooks, so it is its own helper method.
"""
def writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum):
    #this method writes all parameters to a file.
    #size	iterations     boolLSTM	boolAdam	boolNesterov	dropout	kernel	pool	strideC	strideP	momentum	decay	learning rate	beta1	beta2	epsilon	amsgrad
    file.write("grid search iteration: " + str(searchNum) + "\n")
    file.write("neuronLayer " + " ".join(str(x) for x in neuronLayer) + "\n")
    file.write("iterations "+ str(iterations) + "\n")
    file.write("boolLSTM "+ str(boolLSTM) + "\n")
    file.write("boolAdam "+ str(boolAdam) + "\n")
    file.write("boolNest "+ str(boolNest) + "\n")
    file.write("drop " + str(drop) + "\n")
    file.write("kernel " + " ".join(str(x) for x in kernel) + "\n")
    file.write("pool " + " ".join(str(x) for x in pool) + "\n")
    file.write("strideC " + " ".join(str(x) for x in strideC) + "\n")
    file.write("strideP " + " ".join(str(x) for x in strideP) + "\n")
    file.write("momentum "+ str(momentum) + "\n")
    file.write("decay "+ str(decay) + "\n")
    file.write("learnRate " + str(learnRate) + "\n")
    file.write("beta1 " + str(b1) + "\n")
    file.write("beta2 " + str(b2) + "\n")
    file.write("epsilon " + str(epsilon) + "\n")
    file.write("amsgrad " + str(amsgrad) + "\n")



"""
Given two different filenames, load in data and labels for a data set (training, development, or validation).
"""
def load_data(data_filename, labels_filename):
    #open data
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)
    
    #open labels
    with open(Y_train_filename, 'rb') as i:
       labels = pickle.load(i)

    return data, labels
