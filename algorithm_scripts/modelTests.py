'''
script for running pretrained models with the final test data
'''

#imports
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
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
import pickle
import os
import datetime
import time
import random
import sys


#will allow for files to have their text treated as text in illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def makePlots(output, modelName, fpr_test, tpr_test,  test_pred):
    '''
    this method creates all relevent metric plots.

    Arguments:
        model_hist : History object created from a model.fit call
        output : beginning of file name for each plot image output
        fpr_train, tpr_train, fpr_dev, tpr_dev : true/false positives for train/dev datasets respectively
    Returns:
        nothing. should create plot images
    '''

    #roc plot
    plt.plot([0,1], [0,1], 'r--', label = '0.5 line')
    plt.plot(fpr_test, tpr_test, label='test area = {:.3f}'.format(skm.auc(fpr_test,tpr_test)))
    #plt.ylim(-1, 1)
    plt.legend()
    plt.ylabel("True positive")
    plt.xlabel("False positives")
    plt.title(modelName + " ROC")
    plt.savefig(output + "_roc.pdf", format="pdf")
    plt.cla()

    #training confusion matrix
    test_class = np.round(test_pred)
    for x in test_class:
        int(x)

    cm_test = skm.confusion_matrix(test_label, test_class)

    xlabel = 'Predicted labels'
    ylabel = 'True labels'
    test_title = 'Testset ROC Confusion Matrix'


    ax = sns.heatmap(cm_test, annot=True, fmt='d', cbar=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(test_title)
    ax.figure.savefig(output + '_test_cm.pdf', format='pdf')
    plt.cla()
    ax.clear()


#path for data output. each file should contain all used params after training and metrics
outputDir = "./data/testset/"
X_val_filename = '/glade/work/joshuadr/IPython/30_lead/X_val/X_val.txt'

Y_val_filename = '/glade/work/joshuadr/IPython/30_lead/Y_val/station1/Y_val.txt'


with open(X_val_filename, 'rb') as h:
    test_data = pickle.load(h)

with open(Y_val_filename, 'rb') as k:
    test_label = pickle.load(k)

#reshape all data files.
test_data2 = test_data.reshape(-1,120,340,1)

'''
 ###exampe ###
rnn30 = load_model("./data/Recur/lrE2/180716__30_0_rnn.h5")

rnn30_test_pred = rnn30.predict(test_data2).ravel()
rnn30_fpr_test, rnn30_tpr_test, rnn30_thresholds_test = skm.roc_curve(test_label,rnn30_test_pred)

rnn30_file = open(outputDir + originalfilename + '.txt', "w+")

rnn30_score = rnn30.evaluate(test_data2, test_label, verbose=1)
rnn30_file.write("%s: %.2f%%" % (rnn30.metrics_names[1], score[1]*100))

makePlots(outputDir + originalfilename, "RNN 30", rnn30_fpr_test, rnn30_tpr_test)

'''

''' RNN tests!'''
#rnn 20 lead
rnn20 = load_model("./data/Recur/lrE2/180716__20_0_rnn.h5")

rnn20_test_pred = rnn20.predict(test_data2).ravel()
rnn20_fpr_test, rnn20_tpr_test, rnn20_thresholds_test = skm.roc_curve(test_label,rnn20_test_pred)
rnn20_auroc = skm.auc(rnn20_fpr_test,rnn20_tpr_test)

rnn20_score = rnn20.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__20_0_rnn", "RNN 20", rnn20_fpr_test, rnn20_tpr_test, rnn20_test_pred)

rnn20_file = open(outputDir + "180716__20_0_rnn.txt", "w+")

rnn20_file.write("%s: %.2f%%\n" % (rnn20.metrics_names[1], rnn20_score[1]*100))
rnn20_file.write("%s: %.2f%%" % ("AUROC score", rnn20_auroc))
rnn20_file.write("\n\n")
with redirect_stdout(rnn20_file):
    rnn20.summary()

#rnn 30 lead
rnn30 = load_model("./data/Recur/lrE2/180716__30_0_rnn.h5")

rnn30_test_pred = rnn30.predict(test_data2).ravel()
rnn30_fpr_test, rnn30_tpr_test, rnn30_thresholds_test = skm.roc_curve(test_label,rnn30_test_pred)
rnn30_auroc = skm.auc(rnn30_fpr_test,rnn30_tpr_test)

rnn30_score = rnn30.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__30_0_rnn", "RNN 30", rnn30_fpr_test, rnn30_tpr_test, rnn30_test_pred)

rnn30_file = open(outputDir + "180716__30_0_rnn.txt", "w+")

rnn30_file.write("%s: %.2f%%\n" % (rnn30.metrics_names[1], rnn30_score[1]*100))
rnn30_file.write("%s: %.2f%%" % ("AUROC score", rnn30_auroc))
rnn30_file.write("\n\n")
with redirect_stdout(rnn30_file):
    rnn30.summary()

#rnn 40 lead
rnn40 = load_model("./data/Recur/lrE2/180716__40_0_rnn.h5")

rnn40_test_pred = rnn40.predict(test_data2).ravel()
rnn40_fpr_test, rnn40_tpr_test, rnn40_thresholds_test = skm.roc_curve(test_label,rnn40_test_pred)
rnn40_auroc = skm.auc(rnn40_fpr_test,rnn40_tpr_test)

rnn40_score = rnn40.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__40_0_rnn", "RNN 40", rnn40_fpr_test, rnn40_tpr_test, rnn40_test_pred)

rnn40_file = open(outputDir + "180716__40_0_rnn.txt", "w+")

rnn40_file.write("%s: %.2f%%\n" % (rnn40.metrics_names[1], rnn40_score[1]*100))
rnn40_file.write("%s: %.2f%%" % ("AUROC score", rnn40_auroc))
rnn40_file.write("\n\n")
with redirect_stdout(rnn40_file):
    rnn40.summary()

#rnn 50 lead
rnn50 = load_model("./data/Recur/lrE2/180716__50_0_rnn.h5")

rnn50_test_pred = rnn50.predict(test_data2).ravel()
rnn50_fpr_test, rnn50_tpr_test, rnn50_thresholds_test = skm.roc_curve(test_label,rnn50_test_pred)
rnn50_auroc = skm.auc(rnn50_fpr_test,rnn50_tpr_test)

rnn50_score = rnn50.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__50_0_rnn", "RNN 50", rnn50_fpr_test, rnn50_tpr_test, rnn50_test_pred)

rnn50_file = open(outputDir + "180716__50_0_rnn.txt", "w+")

rnn50_file.write("%s: %.2f%%\n" % (rnn50.metrics_names[1], rnn50_score[1]*100))
rnn50_file.write("%s: %.2f%%" % ("AUROC score", rnn50_auroc))
rnn50_file.write("\n\n")
with redirect_stdout(rnn50_file):
    rnn50.summary()

''' CNN tests!'''

#cnn 20 lead
cnn20 = load_model("./data/Recur/lrE2/180716__20_0_cnn.h5")

cnn20_test_pred = cnn20.predict(test_data2).ravel()
cnn20_fpr_test, cnn20_tpr_test, cnn20_thresholds_test = skm.roc_curve(test_label,cnn20_test_pred)
cnn20_auroc = skm.auc(cnn20_fpr_test,cnn20_tpr_test)

cnn20_score = cnn20.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__20_0_cnn", "RNN 20", cnn20_fpr_test, cnn20_tpr_test, cnn20_test_pred)

cnn20_file = open(outputDir + "180716__20_0_cnn.txt", "w+")

cnn20_file.write("%s: %.2f%%\n" % (cnn20.metrics_names[1], cnn20_score[1]*100))
cnn20_file.write("%s: %.2f%%" % ("AUROC score", cnn20_auroc))
cnn20_file.write("\n\n")
with redirect_stdout(cnn20_file):
    cnn20.summary()

#cnn 30 lead
cnn30 = load_model("./data/Recur/lrE2/180716__30_0_cnn.h5")

cnn30_test_pred = cnn30.predict(test_data2).ravel()
cnn30_fpr_test, cnn30_tpr_test, cnn30_thresholds_test = skm.roc_curve(test_label,cnn30_test_pred)
cnn30_auroc = skm.auc(cnn30_fpr_test,cnn30_tpr_test)

cnn30_score = cnn30.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__30_0_cnn", "RNN 30", cnn30_fpr_test, cnn30_tpr_test, cnn30_test_pred)

cnn30_file = open(outputDir + "180716__30_0_cnn.txt", "w+")

cnn30_file.write("%s: %.2f%%\n" % (cnn30.metrics_names[1], cnn30_score[1]*100))
cnn30_file.write("%s: %.2f%%" % ("AUROC score", cnn30_auroc))
cnn30_file.write("\n\n")
with redirect_stdout(cnn30_file):
    cnn30.summary()

#cnn 40 lead
cnn40 = load_model("./data/Recur/lrE2/180716__40_0_cnn.h5")

cnn40_test_pred = cnn40.predict(test_data2).ravel()
cnn40_fpr_test, cnn40_tpr_test, cnn40_thresholds_test = skm.roc_curve(test_label,cnn40_test_pred)
cnn40_auroc = skm.auc(cnn40_fpr_test,cnn40_tpr_test)

cnn40_score = cnn40.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__40_0_cnn", "RNN 40", cnn40_fpr_test, cnn40_tpr_test, cnn40_test_pred)

cnn40_file = open(outputDir + "180716__40_0_cnn.txt", "w+")

cnn40_file.write("%s: %.2f%%\n" % (cnn40.metrics_names[1], cnn40_score[1]*100))
cnn40_file.write("%s: %.2f%%" % ("AUROC score", cnn40_auroc))
cnn40_file.write("\n\n")
with redirect_stdout(cnn40_file):
    cnn40.summary()

#cnn 50 lead
cnn50 = load_model("./data/Recur/lrE2/180716__50_0_cnn.h5")

cnn50_test_pred = cnn50.predict(test_data2).ravel()
cnn50_fpr_test, cnn50_tpr_test, cnn50_thresholds_test = skm.roc_curve(test_label,cnn50_test_pred)
cnn50_auroc = skm.auc(cnn50_fpr_test,cnn50_tpr_test)

cnn50_score = cnn50.evaluate(test_data2, test_label, verbose=1)

makePlots(outputDir + "180716__50_0_cnn", "RNN 50", cnn50_fpr_test, cnn50_tpr_test, cnn50_test_pred)

cnn50_file = open(outputDir + "180716__50_0_cnn.txt", "w+")

cnn50_file.write("%s: %.2f%%\n" % (cnn50.metrics_names[1], cnn50_score[1]*100))
cnn50_file.write("%s: %.2f%%" % ("AUROC score", cnn50_auroc))
cnn50_file.write("\n\n")
with redirect_stdout(cnn50_file):
    cnn50.summary()
