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
Create an ROC curve for the training and development sets. For this, we need arrays that contain the false positive rate and true positive rate, as well as a filename that we can write to.
"""
def rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev):
    #method for saving ROC values to a file for accessing later if needed in excel
    rfile.write("fpr_train\t tpr_train\t fpr_dev\t tpr_dev\n")

    rocVal = [[len(fpr_train)]]

    for val in range(len(fpr_train) - 1):
        #go through train values and save to rocVal
        temp = []
        temp.append(fpr_train[val])
        temp.append(tpr_train[val])
        rocVal.append(temp)

    for val in range(len(fpr_dev) - 1):
        #go thru dev values
        rocVal[val].append(fpr_dev[val])
        rocVal[val].append(tpr_dev[val])

    for t in range(len(rocVal)-1):
        rfile.write(" ".join(str(x)+ "\t" for x in rocVal[t]) + "\n")




"""
This method creates all relevent metric plots.

Arguments:
model_hist : History object created from a model.fit call
output : beginning of file name for each plot image output
fpr_train, tpr_train, fpr_dev, tpr_dev : true/false positives for train/dev datasets respectively
Returns:
nothing. should create plot images
"""
def makePlots(model_hist, output, modelName, fpr_train, tpr_train, fpr_dev, tpr_dev, train_pred, dev_pred):

    #for decerasing the number of tick marks on the grapphs for readibility
    xList = []
    for e in range(len(model_hist.epoch) + 1):
        if e % 25 == 0:
            xList.append(e)
    #bss plot
    plt.plot(model_hist.epoch, model_hist.history["val_loss"], label="validation")
    plt.plot(model_hist.epoch, model_hist.history["loss"], label="train")
    plt.xticks(xList)
    plt.legend()
    plt.ylabel("Loss - Binary Crossentropy")
    plt.xlabel("Epoch")
    plt.title(modelName + " Loss")
    plt.savefig(output + "_loss.pdf", format="pdf")
    plt.cla()

    #accuracy plot
    plt.plot(model_hist.epoch, model_hist.history["val_binary_accuracy"], label="validation")
    plt.plot(model_hist.epoch, model_hist.history["binary_accuracy"], label="train")
    plt.xticks(xList)
    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("Epoch")
    plt.title(modelName + " Accuracy")
    plt.savefig(output + "_accuracy.pdf", format="pdf")
    plt.cla()

    #roc plot
    plt.plot([0,1], [0,1], 'r--', label = '0.5 line')
    plt.plot(fpr_dev, tpr_dev, label='validation area = {:.3f})'.format(skm.auc(fpr_dev,tpr_dev)))
    plt.plot(fpr_train, tpr_train, label='train area = {:.3f}'.format(skm.auc(fpr_train,tpr_train)))
    plt.legend()
    plt.ylabel("True positive")
    plt.xlabel("False positives")
    plt.title(modelName + " ROC")
    plt.savefig(output + "_roc.pdf", format="pdf")
    plt.cla()
    
    #training confusion matrix
    train_class = np.round(train_pred)
    for x in train_class:
        int(x)

    cm_train = skm.confusion_matrix(train_label, train_class)

    xlabel = 'Predicted labels'
    ylabel = 'True labels'
    train_title = 'Training ROC Confusion Matrix'

    #use seaborn's sns.heatmap() function for pretty plotting of confusion matrix
    ax = sns.heatmap(cm_train, annot=True, fmt='d', cbar=False)

    #set x and y labels, as well as title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(train_title)
    ax.figure.savefig(output + '_train_cm.pdf', format='pdf')
    plt.cla()
    ax.clear()
    
    #dev confusion matrix
    dev_class = np.round(dev_pred)
    for y in dev_class:
        int(y)
        
    cm_dev = skm.confusion_matrix(dev_label, dev_class)
    dev_title = 'Dev Set ROC Confusion Matrix'

    #set x and y labels, as well as title
    ax2 = sns.heatmap(cm_dev, annot=True, fmt='d', cbar=False)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(dev_title)
    ax2.figure.savefig(output + '_dev_cm.pdf', format='pdf')
    plt.cla()
    ax2.clear()


