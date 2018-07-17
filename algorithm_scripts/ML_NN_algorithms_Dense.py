"""
"""
from contextlib import redirect_stdout
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
outputDir = "./data/Dense/final/"

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)

#hyperparameters and paramters
#SGD parameters
dropout = 0.5
momentum = 0.99

learningRate = 0.154
epochs = 219
decay = 1.02e-06
batch = 175

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


def makePlots(model_hist, output, modelName, fpr_train, tpr_train, fpr_dev, tpr_dev):
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

#dense nn
    # based off of dnn from Negin. just need to focus on optimizing
def dnn(neuronLayer, drop, learnRate, momentum, decay,boolAdam, boolNest, b1, b2, epsilon, amsgrad,iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum, batch):
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
    print("dense neural network")
    #set final output name/location.
    outputFile = outputDL + "dnn"
    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file, neuronLayer, iterations, None, boolAdam, boolNest, drop, [None], [None], [None], [None], momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum)

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

    dense_hist = denseModel.fit(train_data, train_label, batch_size=batch, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = denseModel.predict(train_data).ravel()
    dev_pred = denseModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rfile.write("fpr_train\t tpr_train\t fpr_dev\t tpr_dev\n")

    for val in range(len(fpr_train) - 1):
        for val in range(len(fpr_dev) - 1):
            rfile.write(str(fpr_train[val]) + "\t" + str(tpr_train[val]) + "\t" + str(fpr_dev[val]) + "\t" + str(tpr_dev[val]) + "\n")

    makePlots(dense_hist, outputFile, "Dense Neural Net",fpr_train, tpr_train, fpr_dev, tpr_dev)
    
    denseModel.save(outputFile+ '.h5')

    return denseModel, skm.auc(fpr_dev,tpr_dev)


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

    X_train_filename = '/glade/work/joshuadr/IPython/X/30_lead/X_train/X_train.txt'
    X_dev_filename = '/glade/work/joshuadr/IPython/X/30_lead/X_dev/X_dev.txt'
    X_val_filename = '/glade/work/joshuadr/IPython/X/30_lead/X_val/X_val.txt'

    Y_train_filename = '/glade/work/joshuadr/IPython/Y/Y_train/station0/Y_train.txt'
    Y_dev_filename = '/glade/work/joshuadr/IPython/Y/Y_dev/station0/Y_dev.txt'
    Y_val_filename = '/glade/work/joshuadr/IPython/Y/Y_val/station0/Y_val.txt'

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
    #bfile.write("learningRates tried: " + " ".join(str(x) for x in learningRate) + "\n")

    #best scores for each net and the associated parameters
    #will also have to change the param lists depending on which params are being optimized
    bestDnnAUROC = 0
    bestDnnSearchNum = 0


    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    start = time.time()
    i = 0
    try_this = 20

    #train models with grid search
    for j in np.arange(try_this):
        outputSearch = outputFile + str(i) + "_"

        denseNN, dnnAUROC = dnn([16,16], dropout, learningRate, momentum, decay, boolNest, boolAdam, beta_1, beta_2, epsilon, amsgrad,epochs,train_data2, train_label,dev_data2, dev_label, outputSearch, i, batch) #these are all negins values right now.
        if dnnAUROC > bestDnnAUROC:
            bestDnnAUROC = dnnAUROC
            bestDnnSearchNum = i

        i += 1


    bfile.write("best DNN AUROC for dev set: " + str(bestDnnAUROC) + "\n")
    bfile.write("best DNN search iteration for dev set: " + str(bestDnnSearchNum) + "\n")


    print("runtime ",time.time() - start, " seconds")
    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
