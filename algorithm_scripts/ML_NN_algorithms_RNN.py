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
outputDir = "./data/Recur/lrE/"

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)

#hyperparameters and paramters
#SGD parameters
dropout = 0.5
learningRate = [0.001, 0.01, 0.05, 0.1, 0.5]
momentum = 0.99
decay = 1e-4
boolNest = True

epochs = [150, 200, 250]

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
    plt.savefig(output + '_loss.png')
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
def dnn(neuronLayer, drop, learnRate, momentum, decay,boolAdam, boolNest, b1, b2, epsilon, amsgrad,iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum):
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

    dense_hist = denseModel.fit(train_data, train_label, batch_size=256, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = denseModel.predict(train_data).ravel()
    dev_pred = denseModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(dense_hist, outputFile, "Dense Neural Net",fpr_train, tpr_train, fpr_dev, tpr_dev)

    return denseModel, skm.auc(fpr_dev,tpr_dev)

#cnn
    # start with lenet
def cnn(neuronLayer, kernel, pool,strideC, strideP, drop, learnRate, momentum, decay,boolNest,boolAdam, b1, b2, epsilon, amsgrad,iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum):
    '''
    implements a convolutional neural network and creates files with parameters and plots

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
        alexnet: https://gist.github.com/JBed/c2fb3ce8ed299f197eff
    '''
    print("convoultional neural network")

    #make sure all lists are the same length s.t. the for loops for setting up dont break
    assert (len(kernel) == len(strideC))
    assert (len(pool) == len(strideP))

    outputFile = outputDL + "cnn"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,neuronLayer, iterations, None, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum)

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
    conv_hist = convModel.fit(train_data, train_label,batch_size=256,epochs=iterations,verbose=1,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = convModel.predict(train_data).ravel()
    dev_pred = convModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(conv_hist, outputFile, "Conv Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    return convModel, skm.auc(fpr_dev,tpr_dev)
# rnn
    # do stuff. look at what might be a good starting point; could try LSTM??
def rnn(neuronLayer, kernel, pool, strideC, strideP, drop, learnRate, momentum, decay,boolNest,boolLSTM, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum):
    '''
    implements a recurrent neural network and creates files with parameters and plots

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

    Example:
        rnn([],)
    '''
    print("recurrent neural network")
    outputFile = outputDL + "rnn"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum)

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
    recur_hist = recurModel.fit(train_data, train_label,batch_size=256,epochs=iterations,verbose=1,validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = recurModel.predict(train_data).ravel()
    dev_pred = recurModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(recur_hist, outputFile, "LSTM Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    return recurModel, skm.auc(fpr_dev,tpr_dev)

def alex(learnRate, momentum, decay, boolNest, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputSearch, searchNum):

    outputFile = outputDL + "alex"

    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    writeFile(file,[4096, 4096, 1000], 1, None, True, False, 0.4, [11, 11, 3,3,3], [2,2,1], [1,1,1,1], [2,2,1], None, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum)

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
    alex_hist = model.fit(train_data, train_label, batch_size=64, epochs=iterations, verbose=1, validation_data=(dev_data, dev_label))

    #calculate ROC info
    train_pred = model.predict(train_data).ravel()
    dev_pred = model.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    makePlots(alex_hist, outputFile, "Alex Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

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

    X_train_filename = '/glade/work/joshuadr/IPython/X/20_lead/X_train/X_train.txt'
    X_dev_filename = '/glade/work/joshuadr/IPython/X/20_lead/X_dev/X_dev.txt'
    X_val_filename = '/glade/work/joshuadr/IPython/X/20_lead/X_val/X_val.txt'

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
    bfile.write("epochs tried: " + " ".join(str(x) for x in epochs) + "\n")
    #bfile.write("dropouts tried: " + " ".join(str(x) for x in dropout) + "\n")
    bfile.write("learningRates tried: " + " ".join(str(x) for x in learningRate) + "\n")

    #best scores for each net and the associated parameters
    #will also have to change the param lists depending on which params are being optimized
    bestDnnAUROC = 0
    bestDnnParams = [epochs[0], learningRate[0]]
    bestDnnSearchNum = 0
    bestCnnAUROC = 0
    bestCnnParams = [epochs[0], learningRate[0]]
    bestCnnSearchNum = 0
    bestRnnAUROC = 0
    bestRnnParams = [epochs[0], learningRate[0]]
    bestRnnSearchNum = 0

    bestAlexAUROC = 0
    bestAlexParams = [epochs[0], learningRate[0]]
    bestAlexSearchNum = 0


    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    start = time.time()
    i = 0

    #train models with grid search
    for e in epochs:
        for l in learningRate:
            outputSearch = outputFile + str(i) + "_"

            recurrNN, rnnAUROC = rnn([20,60],kernel, pool, strideC, strideP, dropout, l, momentum, decay, boolNest,True, boolAdam,beta_1, beta_2, epsilon, amsgrad, e, train_data2, train_label, dev_data2, dev_label,outputSearch, i)
            if rnnAUROC > bestRnnAUROC:
                bestRnnAUROC = rnnAUROC
                bestRnnParams = [e, l]
                bestRnnSearchNum = i

            i += 1


    bfile.write("best RNN AUROC for dev set: " + str(bestRnnAUROC) + "\n")
    bfile.write("best RNN search iteration for dev set: " + str(bestRnnSearchNum) + "\n")
    bfile.write("best parameters for RNN: " + " ".join(str(x) for x in bestRnnParams) + "\n\n")


    print("runtime ",time.time() - start, " seconds")
    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
