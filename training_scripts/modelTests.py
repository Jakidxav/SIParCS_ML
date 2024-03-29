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

#path for data output. each file should contain all used params after training and metrics
outputDir = "./data/testset/"
X_val_filename20 = '/glade/work/jakidxav/IPython/X/20_lead/X_val/X_val.txt'
X_val_filename30 = '/glade/work/jakidxav/IPython/X/30_lead/X_val/X_val.txt'
X_val_filename40 = '/glade/work/jakidxav/IPython/X/40_lead/X_val/X_val.txt'
X_val_filename50 = '/glade/work/jakidxav/IPython/X/50_lead/X_val/X_val.txt'

Y_val_filename = '/glade/work/jakidxav/IPython/Y/Y_val/station0/Y_val.txt'


with open(X_val_filename20, 'rb') as h:
    test_data20 = pickle.load(h)

with open(X_val_filename30, 'rb') as h:
    test_data30 = pickle.load(h)

with open(X_val_filename40, 'rb') as h:
    test_data40 = pickle.load(h)

with open(X_val_filename50, 'rb') as h:
    test_data50 = pickle.load(h)

with open(Y_val_filename, 'rb') as k:
    test_label = pickle.load(k)

#reshape all data files.
test_data20_2 = test_data20.reshape(-1,120,340,1)
test_data30_2 = test_data30.reshape(-1,120,340,1)
test_data40_2 = test_data40.reshape(-1,120,340,1)
test_data50_2 = test_data50.reshape(-1,120,340,1)

#plots roc with the one week predictive metric
week_test_label = test_label.copy()

max = len(test_label) - 1
print('changed here\n')
for x in range(max):
    #if index does not contain a true label then look for surrounding labels
    if test_label[x] != 1:
        #less than 3 values before x in array.
        if x <= (max - 4):
            if 1 in test_label[x:x + 4]:
                week_test_label[x] = 1
        if x >= (4):
            if 1 in test_label[x - 3:x]:
                week_test_label[x] = 1
            x += 1


""" RNN tests! """
#rnn 20 lead
rnn20 = load_model("./best_models/180725__20_8.4_rnn.h5")

rnn20_test_pred = rnn20.predict(test_data20_2).ravel()
rnn20_test_pred_round = np.where(rnn20_test_pred >= 0.5 , 1, 0)
rnn20_fpr_test, rnn20_tpr_test, rnn20_thresholds_test = skm.roc_curve(test_label,rnn20_test_pred_round)
rnn20_auroc = skm.auc(rnn20_fpr_test,rnn20_tpr_test)

rnn20_score = rnn20.evaluate(test_data20_2, test_label, verbose=1)

makePlots(outputDir + "180716__20_0_rnn", "RNN 20", rnn20_fpr_test, rnn20_tpr_test, rnn20_test_pred)

#1 week prediction look at roc
week_rnn20_fpr_test, week_rnn20_tpr_test, week_rnn20_thresholds_test = skm.roc_curve(week_test_label,rnn20_test_pred_round)
week_rnn20_auroc = skm.auc(week_rnn20_fpr_test,week_rnn20_tpr_test)

makePlots(outputDir + "180716__20_0_rnn_week", "RNN 20", week_rnn20_fpr_test, week_rnn20_tpr_test, rnn20_test_pred)

# text file output
rnn20_file = open(outputDir + "180716__20_0_rnn.txt", "w+")

rnn20_file.write("%s: %.2f%%\n" % (rnn20.metrics_names[1], rnn20_score[1]*100))
rnn20_file.write("%s: %.2f%%" % ("AUROC score", rnn20_auroc))
rnn20_file.write("%s: %.2f%%" % ("AUROC score week time ", rnn20_auroc))
rnn20_file.write("\n\n")
with redirect_stdout(rnn20_file):
    rnn20.summary()

#rnn 30 lead
rnn30 = load_model("./best_models/180725__30_9.1_rnn.h5")

rnn30_test_pred = rnn20.predict(test_data30_2).ravel()
rnn30_test_pred_round = np.where(rnn30_test_pred >= 0.5 , 1, 0)
rnn30_fpr_test, rnn30_tpr_test, rnn30_thresholds_test = skm.roc_curve(test_label,rnn30_test_pred_round)
rnn30_auroc = skm.auc(rnn30_fpr_test,rnn30_tpr_test)

rnn30_score = rnn30.evaluate(test_data30_2, test_label, verbose=1)

makePlots(outputDir + "180716__30_0_rnn", "RNN 30", rnn30_fpr_test, rnn30_tpr_test, rnn30_test_pred)

#1 week prediction look at roc
week_rnn30_fpr_test, week_rnn30_tpr_test, week_rnn30_thresholds_test = skm.roc_curve(week_test_label,rnn30_test_pred_round)
week_rnn30_auroc = skm.auc(week_rnn30_fpr_test,week_rnn30_tpr_test)

makePlots(outputDir + "180716__30_0_rnn_week", "RNN 30", week_rnn30_fpr_test, week_rnn30_tpr_test, rnn30_test_pred)

# text file output
rnn30_file = open(outputDir + "180716__30_0_rnn.txt", "w+")

rnn30_file.write("%s: %.2f%%\n" % (rnn30.metrics_names[1], rnn30_score[1]*100))
rnn30_file.write("%s: %.2f%%" % ("AUROC score", rnn30_auroc))
rnn30_file.write("%s: %.2f%%" % ("AUROC score week time ", rnn30_auroc))
rnn30_file.write("\n\n")
with redirect_stdout(rnn30_file):
    rnn30.summary()

#rnn 40 lead
rnn40 = load_model("./best_models/180725__40_9.0_rnn.h5")

rnn40_test_pred = rnn40.predict(test_data40_2).ravel()
rnn40_test_pred_round = np.where(rnn40_test_pred >= 0.5 , 1, 0)
rnn40_fpr_test, rnn40_tpr_test, rnn40_thresholds_test = skm.roc_curve(test_label,rnn40_test_pred_round)
rnn40_auroc = skm.auc(rnn40_fpr_test,rnn40_tpr_test)

rnn40_score = rnn40.evaluate(test_data40_2, test_label, verbose=1)

makePlots(outputDir + "180716__40_0_rnn", "RNN 40", rnn40_fpr_test, rnn40_tpr_test, rnn40_test_pred)

#1 week prediction look at roc
week_rnn40_fpr_test, week_rnn40_tpr_test, week_rnn40_thresholds_test = skm.roc_curve(week_test_label,rnn40_test_pred_round)
week_rnn40_auroc = skm.auc(week_rnn40_fpr_test,week_rnn40_tpr_test)

makePlots(outputDir + "180716__40_0_rnn_week", "RNN 40", week_rnn40_fpr_test, week_rnn40_tpr_test, rnn40_test_pred)

# text file output
rnn40_file = open(outputDir + "180716__40_0_rnn.txt", "w+")

rnn40_file.write("%s: %.2f%%\n" % (rnn40.metrics_names[1], rnn40_score[1]*100))
rnn40_file.write("%s: %.2f%%" % ("AUROC score", rnn40_auroc))
rnn40_file.write("%s: %.2f%%" % ("AUROC score week time ", rnn40_auroc))
rnn40_file.write("\n\n")
with redirect_stdout(rnn40_file):
    rnn40.summary()

#rnn 50 lead
rnn50 = load_model("./best_models/180725__50_1.1_rnn.h5")

rnn50_test_pred = rnn50.predict(test_data50_2).ravel()
rnn50_test_pred_round = np.where(rnn50_test_pred >= 0.5 , 1, 0)
rnn50_fpr_test, rnn50_tpr_test, rnn50_thresholds_test = skm.roc_curve(test_label,rnn50_test_pred_round)
rnn50_auroc = skm.auc(rnn50_fpr_test,rnn50_tpr_test)

rnn50_score = rnn50.evaluate(test_data50_2, test_label, verbose=1)

makePlots(outputDir + "180716__50_0_rnn", "RNN 50", rnn50_fpr_test, rnn50_tpr_test, rnn50_test_pred)

#1 week prediction look at roc
week_rnn50_fpr_test, week_rnn50_tpr_test, week_rnn50_thresholds_test = skm.roc_curve(week_test_label,rnn50_test_pred_round)
week_rnn50_auroc = skm.auc(week_rnn50_fpr_test,week_rnn50_tpr_test)

makePlots(outputDir + "180716__50_0_rnn_week", "RNN 50", week_rnn50_fpr_test, week_rnn50_tpr_test, rnn50_test_pred)

# text file output
rnn50_file = open(outputDir + "180716__50_0_rnn.txt", "w+")

rnn50_file.write("%s: %.2f%%\n" % (rnn50.metrics_names[1], rnn50_score[1]*100))
rnn50_file.write("%s: %.2f%%" % ("AUROC score", rnn50_auroc))
rnn50_file.write("%s: %.2f%%" % ("AUROC score week time ", rnn50_auroc))
rnn50_file.write("\n\n")
with redirect_stdout(rnn50_file):
    rnn50.summary()


''' CNN tests!'''

#rnn 20 lead
cnn20 = load_model("./best_models/180726__20_0_cnn.h5")

cnn20_test_pred = cnn20.predict(test_data20_2).ravel()
cnn20_test_pred_round = np.where(cnn20_test_pred >= 0.5 , 1, 0)
cnn20_fpr_test, cnn20_tpr_test, cnn20_thresholds_test = skm.roc_curve(test_label,cnn20_test_pred_round)
cnn20_auroc = skm.auc(cnn20_fpr_test,cnn20_tpr_test)

cnn20_score = cnn20.evaluate(test_data20_2, test_label, verbose=1)

makePlots(outputDir + "180726__20_0_cnn", "cnn 20", cnn20_fpr_test, cnn20_tpr_test, cnn20_test_pred)

#1 week prediction look at roc
week_cnn20_fpr_test, week_cnn20_tpr_test, week_cnn20_thresholds_test = skm.roc_curve(week_test_label,cnn20_test_pred_round)
week_cnn20_auroc = skm.auc(week_cnn20_fpr_test,week_cnn20_tpr_test)

makePlots(outputDir + "180726__20_0_cnn_week", "cnn 20", week_cnn20_fpr_test, week_cnn20_tpr_test, cnn20_test_pred)

# text file output
cnn20_file = open(outputDir + "180726__20_0_cnn.txt", "w+")

cnn20_file.write("%s: %.2f%%\n" % (cnn20.metrics_names[1], cnn20_score[1]*100))
cnn20_file.write("%s: %.2f%%" % ("AUROC score", cnn20_auroc))
cnn20_file.write("%s: %.2f%%" % ("AUROC score week time ", cnn20_auroc))
cnn20_file.write("\n\n")
with redirect_stdout(cnn20_file):
    cnn20.summary()

#cnn 30 lead
cnn30 = load_model("./best_models/180725__30_0.4_cnn.h5")

cnn30_test_pred = cnn20.predict(test_data30_2).ravel()
cnn30_test_pred_round = np.where(cnn30_test_pred >= 0.5 , 1, 0)
cnn30_fpr_test, cnn30_tpr_test, cnn30_thresholds_test = skm.roc_curve(test_label,cnn30_test_pred_round)
cnn30_auroc = skm.auc(cnn30_fpr_test,cnn30_tpr_test)

cnn30_score = cnn30.evaluate(test_data30_2, test_label, verbose=1)

makePlots(outputDir + "180725__30_0.4_cnn", "cnn 30", cnn30_fpr_test, cnn30_tpr_test, cnn30_test_pred)

#1 week prediction look at roc
week_cnn30_fpr_test, week_cnn30_tpr_test, week_cnn30_thresholds_test = skm.roc_curve(week_test_label,cnn30_test_pred_round)
week_cnn30_auroc = skm.auc(week_cnn30_fpr_test,week_cnn30_tpr_test)

makePlots(outputDir + "180725__30_0.4_cnn_week", "cnn 30", week_cnn30_fpr_test, week_cnn30_tpr_test, cnn30_test_pred)

# text file output
cnn30_file = open(outputDir + "180725__30_0.4_cnn.txt", "w+")

cnn30_file.write("%s: %.2f%%\n" % (cnn30.metrics_names[1], cnn30_score[1]*100))
cnn30_file.write("%s: %.2f%%" % ("AUROC score", cnn30_auroc))
cnn30_file.write("%s: %.2f%%" % ("AUROC score week time ", cnn30_auroc))
cnn30_file.write("\n\n")
with redirect_stdout(cnn30_file):
    cnn30.summary()

#cnn 40 lead
cnn40 = load_model("./best_models/180725__40_0.0_cnn.h5")

cnn40_test_pred = cnn40.predict(test_data40_2).ravel()
cnn40_test_pred_round = np.where(cnn40_test_pred >= 0.5 , 1, 0)
cnn40_fpr_test, cnn40_tpr_test, cnn40_thresholds_test = skm.roc_curve(test_label,cnn40_test_pred_round)
cnn40_auroc = skm.auc(cnn40_fpr_test,cnn40_tpr_test)

cnn40_score = cnn40.evaluate(test_data40_2, test_label, verbose=1)

makePlots(outputDir + "180725__40_0.0_cnn", "cnn 40", cnn40_fpr_test, cnn40_tpr_test, cnn40_test_pred)

#1 week prediction look at roc
week_cnn40_fpr_test, week_cnn40_tpr_test, week_cnn40_thresholds_test = skm.roc_curve(week_test_label,cnn40_test_pred_round)
week_cnn40_auroc = skm.auc(week_cnn40_fpr_test,week_cnn40_tpr_test)

makePlots(outputDir + "180725__40_0.0_cnn_week", "cnn 40", week_cnn40_fpr_test, week_cnn40_tpr_test, cnn40_test_pred)

# text file output
cnn40_file = open(outputDir + "180725__40_0.0_cnn.txt", "w+")

cnn40_file.write("%s: %.2f%%\n" % (cnn40.metrics_names[1], cnn40_score[1]*100))
cnn40_file.write("%s: %.2f%%" % ("AUROC score", cnn40_auroc))
cnn40_file.write("%s: %.2f%%" % ("AUROC score week time ", cnn40_auroc))
cnn40_file.write("\n\n")
with redirect_stdout(cnn40_file):
    cnn40.summary()

#cnn 50 lead
cnn50 = load_model("./best_models/180725__50_4.2_cnn.h5")

cnn50_test_pred = cnn50.predict(test_data50_2).ravel()
cnn50_test_pred_round = np.where(cnn50_test_pred >= 0.5 , 1, 0)
cnn50_fpr_test, cnn50_tpr_test, cnn50_thresholds_test = skm.roc_curve(test_label,cnn50_test_pred_round)
cnn50_auroc = skm.auc(cnn50_fpr_test,cnn50_tpr_test)

cnn50_score = cnn50.evaluate(test_data50_2, test_label, verbose=1)

makePlots(outputDir + "180725__50_4.2_cnn", "cnn 50", cnn50_fpr_test, cnn50_tpr_test, cnn50_test_pred)

#1 week prediction look at roc
week_cnn50_fpr_test, week_cnn50_tpr_test, week_cnn50_thresholds_test = skm.roc_curve(week_test_label,cnn50_test_pred_round)
week_cnn50_auroc = skm.auc(week_cnn50_fpr_test,week_cnn50_tpr_test)

makePlots(outputDir + "180725__50_4.2_cnn_week", "cnn 50", week_cnn50_fpr_test, week_cnn50_tpr_test, cnn50_test_pred)

# text file output
cnn50_file = open(outputDir + "180725__50_4.2_cnn.txt", "w+")

cnn50_file.write("%s: %.2f%%\n" % (cnn50.metrics_names[1], cnn50_score[1]*100))
cnn50_file.write("%s: %.2f%%" % ("AUROC score", cnn50_auroc))
cnn50_file.write("%s: %.2f%%" % ("AUROC score week time ", cnn50_auroc))
cnn50_file.write("\n\n")
with redirect_stdout(cnn50_file):
    cnn50.summary()
