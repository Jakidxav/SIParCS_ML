"""

@author Negin Sobhani, Jakidxav, Karen Stengel

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
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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

#will allow for files to
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#path for data output. each file should contain all used params after training and metrics
outputDir = "./data/Alex/weights/"

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)
posWeight = [2,4,8,16]
trials = 3
#hyperparameters and paramters
#SGD parameters
dropout = 0.5
learningRate = [0.49,  0.123, 0.225, 0.357, 0.347, 0.123, 0.001, 0.011, 0.184, 0.49,  0.154, 0.032,
 0.205, 0.052, 0.266, 0.388]
momentum = 0.99
decay = 1e-4
boolNest = True

epochs = [213, 178, 250, 198, 233, 176, 158, 233, 156, 229, 219, 211, 182, 207, 247, 209]

#parameters for conv/pooling layers
strideC = [5,5, 1]
strideP = [2,2]
kernel = [5, 5,1]
pool = [2,2]

#parameters for Adam optimizaiton
boolAdam = True #change to false if SGD is desired
beta_1=0.9
beta_2=0.999
epsilon=None
amsgrad=False

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


def makePlots(model_hist, output, modelName, fpr_train, tpr_train, fpr_dev, tpr_dev, train_pred, dev_pred):
    '''
    this method creates all relevent metric plots.

    Arguments:
        model_hist : History object created from a model.fit call
        output : beginning of file name for each plot image output
        fpr_train, tpr_train, fpr_dev, tpr_dev : true/false positives for train/dev datasets respectively
    Returns:
        nothing. should create plot images
    '''
    #for decerasing the number of tick marks on the grapphs for readibility
    xList = []
    for e in range(len(model_hist.epoch) + 1):
        if e % 25 == 0:
            xList.append(e)
    #bss plot
    plt.plot(model_hist.epoch, model_hist.history["val_loss"], label="validation")
    plt.plot(model_hist.epoch, model_hist.history["loss"], label="train")
    plt.xticks(xList)
    #plt.ylim(-1, 1)
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
    #plt.ylim(-1, 1)
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
    #plt.ylim(-1, 1)
    plt.legend()
    plt.ylabel("True positive")
    plt.xlabel("False positives")
    plt.title(modelName + " ROC")
    plt.savefig(output + "_roc.pdf", format="pdf")
    plt.cla()

def writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, batch,posWeight):

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
    file.write("weight for positive prediciton " + str(posWeight) + "\n")


def alex(learnRate, momentum, decay, boolNest, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputSearch, searchNum):

    outputFile = outputDL + "alex"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,[4096, 4096, 1000], 1, None, True, False, 0.4, [11, 11, 3,3,3], [2,2,1], [1,1,1,1], [2,2,1], None, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight)

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
    alex_hist = model.fit(train_data, train_label, batch_size=1, epochs=iterations, verbose=1, validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = model.predict(train_data).ravel()
    dev_pred = model.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    makePlots(alex_hist, outputFile, "Alex Net", fpr_train, tpr_train, fpr_dev, tpr_dev, train_pred, dev_pred)

    model.save(outputFile+ '.h5')

    return model, skm.roc_curve(dev_label, dev_pred)
#main stuff
    #this should read in each dataset and call the NN algorithms.
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
    #netType = sys.argv[1] #can be dense, conv, recur, alex, or all
    #stationNum = sys.argv[2] # the station to be used. should be in the form of station1
    #print(netType)

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d_")
    #print(date)

    #for all dataset directories & all data in each: need to walk through the various dierctories that contain each dataset
    #can change this directory once run location on cheyenne is selected

    outputDL = date + "_30_"
    #print(outputDL)
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
    bfile.write("epochs tried: " + " ".join(str(x) for x in epochs) + "\n")
    #bfile.write("dropouts tried: " + " ".join(str(x) for x in dropout) + "\n")
    bfile.write("learningRates tried: " + " ".join(str(x) for x in learningRate) + "\n")

    #best scores for each net and the associated parameters
    #will also have to change the param lists depending on which params are being optimized
    bestAlexAUROC = 0
    bestAlexParams = [epochs[0], learningRate[0]]
    bestAlexSearchNum = 0


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

            alexNN, alexAUROC = alex([20,60],kernel, pool, strideC, strideP, dropout, 0.123, momentum, 1.0e-4, boolNest,True, boolAdam,beta_1, beta_2, epsilon, amsgrad, 17, train_data2, train_label, dev_data2, dev_label,outputSearch, i, batch, w)
            if alexAUROC > modelAUROC:
                model = alexNN
                modelAUROC = alexAUROC
                bestTry = t
        model.save(outputSearch + str(bestTry)+'.h5')

        if modelAUROC > bestAlexAUROC:
            bestAlexAUROC = modelAUROC
            bestAlexParams = [w]
            bestAlexSearchNum = i
        i += 1


    bfile.write("best Alex AUROC for dev set: " + str(bestAlexAUROC) + "\n")
    bfile.write("best Alex search iteration for dev set: " + str(bestAlexSearchNum) + "\n")
    bfile.write("best parameters for Alex: " + " ".join(str(x) for x in bestAlexParams) + "\n\n")

    print("runtime ",time.time() - start, " seconds")
    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
