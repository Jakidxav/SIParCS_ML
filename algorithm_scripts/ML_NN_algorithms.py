"""

@author Negin Sobhani, Joshua Driscol, Karen Stengel

this script is designed to use keras with Tensorflow as the backendself.
it should be used in conjunction with data_processing.py which should set up/create necessary
datasets from the McKinnon et al. dataself plus any other data added to the set.

this script will take the premade datasets and run them through the various ML NN algorithms
and return ROC scores; can use the binary_accuaracy method in keras as a starting pointself.
This script will also include optimizers so that we can get the best
scores possible.

NEED TO FIGURE OUT HOW TO DO THE FOLLOWING IN KERAS EFFECIENTLY
algorithms to include:
    dense nn - already created by Negin so will just need to try optimizing
    CNN - also already tried by Negin so will need to try optimizing.
    siamese nn - possibly use? need to look into best implementation...
    RNN - not yet tried. will need to set up and OPTIMIZE. use LSTM?
    RBFN - ? maybe use? could allow for an interesing analysis....

will need each algorithm method to take in certain parameters to make optimizing easier.
    keras optimizers are specifically for things like momentum, learning rate, epislon, etc etc
    thus, we will have to manually (or write a method to) change number on neurons and/or layers (and # in each layer)
    will probably want to automate this.
    EACH NN METHOD SHOULD TAKE IN # NEURONS
    AS AN ARRAY WHERE EACH INDEX IS A NONINPUT LAYER, EACH RELEVANT TUNING PARAMETER. should use a for loops when setting up
    layers to go through number of neurons and layers. should make each method as versitile as possible s.t. network optimization
    can be easily optimized. can also add in the hidden activation and output activation if needed.

need to write out all info regarding metrics and paramters to a file. should include the original picture with corresponding label,
    and ROC, bss graphs. will need to name file so that these parameters are easily tracked.

    will also need to save models once they are fully trained/work (hopefully)

need to change train_data.shape to use slicing..... Karen is not good at slicing

datasets:
    should be set up with various lead prediction times. ex: 10, 30, 45, 60 days in advance
    should also use a control set with randomly shuffled labels
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from keras.models import Sequential
import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU, TimeDistributed, LSTM
from keras.layers import Dropout, BatchNormalization
from keras.metrics import binary_accuracy
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
import sklearn.metrics as skm
import tensorflow as tf
import pickle
import os
import datetime

#path for data output. each file should contain all used params after training and metrics
outputDir = "./"

#ROC calculations. will need to use this V datasets on all algorithms
def rocValues(obs, pred):
    '''
    implements an AUROC calculation. also graphs the ROC curve

    Arguments:
        obs : array containing actual labels
        pred : array containing predicted labels

    Returns:
        TP, FP, TN, FN
    '''

    TP, FP, TN, FN = 0

    for i in range(len(pred)):
        if obs[i] == pred[i] == 1:
            TP += 1
        if pred[i] == 1 and pred[i] != obs[i]:
            FP += 1
        if pred[i] == 0 and pred[i] != obs[i]:
            FN += 1
        if obs[i] == pred[i] == 0:
            TN += 1

    return TP, FP, TN, FN
def TP(obs, pred):
    return K.metrics.true_positives(obs, pred)
def FP(obs, pred):
    return K.metrics.false_positives(obs, pred)
#brier score and brier skill score. both methods written by Negin
def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    return 1.0 - brier_score_keras(obs, preds) / climo

def makePlots(model_hist, output):
    '''
    this method creates all relevent metric plots.

    Arguments:
        model_hist : History object created from a model.fit call
        output : beginning of file name for each plot image output
    Returns:
        nothing. should create plot images
    '''
    #bss plot
    plt.plot(model_hist.epoch, model_hist.history["val_brier_skill_score_keras"], label="validation")
    plt.plot(model_hist.epoch, model_hist.history["brier_skill_score_keras"], label="train")
    plt.xticks(model_hist.epoch)
    #plt.ylim(-1, 1)
    plt.legend()
    plt.ylabel("Brier Skill Score")
    plt.xlabel("Epoch")
    plt.title("Dense Net Training History")
    plt.savefig(output + '_bss.png')
    plt.clear()

    #accuracy plot
    plt.plot([0,1], [0,1], 'r--', label = '0.5 line')
    plt.plot(model_hist.epoch, model_hist.history["val_binary_accuracy"], label="validation")
    plt.plot(model_hist.epoch, model_hist.history["binary_accuracy"], label="train")
    plt.xticks(model_hist.epoch)
    #plt.ylim(-1, 1)
    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("Epoch")
    plt.title("Dense Net Training History")
    plt.savefig(output + '_accuracy.png')
    plt.clear()

    #roc plot
    plt.plot([0,1], [0,1], 'r--', label = '0.5 line')
    plt.plot(model_hist.history["val_FP"], model_hist.history["val_TP"], label='validation area = {:.3f})'.format(skm.auc(model_hist.history["val_FP"],model_hist.history["val_TP"])))
    plt.plot(model_hist.history["FP"], model_hist.history["TP"], label='train area = {:.3f}'.format(skm.auc(model_hist.history["FP"],model_hist.history["TP"])))
    #plt.xticks(dense_hist.epoch)
    #plt.ylim(-1, 1)
    plt.legend()
    plt.ylabel("True positive")
    plt.xlabel("False positives")
    plt.title("Dense Net ROC")
    plt.savefig(output + '_roc.png')
    plt.clear()


#dense nn
    # based off of dnn from Negin. just need to focus on optimizing
def dnn(neuronLayer, drop, learnRate, momentum, decay,boolNest, iterations, train_data, train_label, dev_data, dev_label, outputDL):
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
        dnn([16,16,2], 0.5, 0.0001, 0.99, 1e-4, True, train_data, train_label, dev_data, dev_label, outputDL)
    '''
    print("dense neural network")
    #set final output name/location.
    outputFile = outputDL + "dnn"
    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")

    file.write("neuronLayer " + " ".join(str(x) for x in neuronLayer) + "\n")
    file.write("drop " + str(drop) + "\n")
    file.write("learnRate " + str(learnRate) + "\n")
    file.write("momentum "+ str(momentum) + "\n")
    file.write("decay "+ str(decay) + "\n")
    file.write("boolNest "+ str(boolNest) + "\n")
    file.write("iterations "+ str(iterations) + "\n")

    #initilaize model with Sequential()
    print(train_data.shape)

    denseModel = Sequential()
    #add first layers
    denseModel.add(AveragePooling2D(pool_size = (2,2), input_shape = (-1, 120, 340, 1))) # negin used this as the first layer. need to double check syntax
    denseModel.add(Flatten())

    for layer in neuronLayer:
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dropout(drop))
        denseModel.add(Dense(neuronLayer[layer], kernel_regularizer=l2(0.0001), activation = 'relu'))

        #this is the output layer; # neurons should be equal to 1
        if(layer == (len(neuronLayer) - 1)):
            denseModel.add(Dropout(drop))
            denseModel.add(Dense(neuronLayer[layer], kernel_regularizer=l2(0.0001), activation = 'sigmoid'))

    #define optimizer
    opt_dense = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)
    denseModel.summary()

    #compile
    denseModel.compile(opt_dense, "mse", metrics=[brier_skill_score_keras, binary_accuracy, TP, FP])

    dense_hist = denseModel.fit(train_data, train_label, batch_size=256, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label))
    #plot info
    makePlots(dense_hist, outputFile)

    return denseModel

#cnn
    # start with 3x convlayer and poolLayer repeats.
    #WILL NEED TO CHANGE THE KERNEL, POOL, STRIDE PARAMS TO BE LISTS SO THAT ALEXNETS/OTHER NETS
    #WITH VARYING STRIDES ETC CAN BE IMPLEMENTED
def cnn(neuronLayer, kernel, pool,strideC, strideP, learnRate, iterations, train_data, train_label, dev_data, dev_label, outputDL):
    '''
    implements a convolutional neural network and creates files with parameters and plots

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        kernel : array of size of conv kernel
        pool : array of pool_size amount
        strideC : array of lengths of conv stride
        strideP : array of lengths of pool stride
        learnRate : learning rate
        iterations : number of iterations to train the model
        train_data : data to train on (numpy array)
        train_label : labels of training data (numpy array)
        outputDL : path for data output

    Returns:
        convModel : a trained keras convolutional network

    Example:
        cnn([32,64, 1000, 2], [5,5], [2,2], [1,1], [1,1], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
        lenet would be: cnn([20,50,500,2], [5,5], [2,2], [1,1], [2,2], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL)
        alexnet: https://gist.github.com/JBed/c2fb3ce8ed299f197eff
    '''
    print("convoultional neural network")

    outputFile = outputDL + "cnn"
    print(outputFile)
    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    file.write("neuronLayer ", neuronLayer)
    file.write("kernel ", kernel)
    file.write("learnRate ", learnRate)
    file.write("pool ", pool)
    file.write("strideC ", strideC)
    file.write("strideP ", strideP)
    file.write("iterations ", iterations)
    #make sure all lists are the same length s.t. the for loops for setting up dont break
    assert (len(kernel) == len(pool) == len(strideC) == len(strideP))

    #initilaize model with Sequential
    convModel = Sequential()

    #add first conv and pooling layers
    convModel.add(Conv2D(neuronLayer[0], kernel_size=(kernel[0], kernel[0]), strides=(strideC[0], strideC[0]),activation='relu', input_shape=train_data.shape[1:]))
    convModel.add(MaxPooling2D(pool_size=(pool[0], pool[0]), strides=(strideP[0], strideP[0])))

    for layer in range(1, len(neuronLayer) - 3):
        convModel.add(Conv2D(neuronLayer[layer], kernel_size = (kernel[layer],kernel[layer]), activation='relu'))
        convModel.add(MaxPooling2D(pool_size=(pool[layer], pool[layer]), strides=(strideP[layer], strideP[layer])))

    convModel.add(Flatten())
    convModel.add(Dense(neuronLayer[len(neuronLayer) - 2], activation='relu'))
    convModel.add(Dense(neuronLayer[len(neuronLayer) - 1], activation='softmax'))

    convModel.summary()

    convModel.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=learnRate),metrics=[brier_skill_score_keras, binary_accuracy, TP, FP])
    conv_hist = convModel.fit(train_data, train_label,batch_size=256,epochs=iterations,verbose=1,validation_data=(dev_data, dev_label))

    #plot stuff
    makePlots(conv_hist, outputFile)

    return convModel
# rnn
    # do stuff. look at what might be a good starting point; could try LSTM??
def rnn(neuronLayer, kernel, pool, strideC, strideP, drop, boolLSTM, iterations, train_data, train_label, dev_data, dev_label, outputDL):
    '''
    implements a recurrent neural network and creates files with parameters and plots

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        kernel : sixe of convolutional kernel
        pool : size of pool
        strideC : size of conv stride
        strideP : size of pooling stride
        dropout : % of neurons to dropout. set to 0 if not using dropout
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
    print(outputFile)
    #create and fill file with parameters and network info
    file = open(outputFile + '.txt', "w+")
    file.write("neuronLayer ", neuronLayer)
    file.write("dropout ", drop)
    file.write("stuff")
    file.write("boolLSTM ", boolLSTM)
    file.write("iterations ", iterations)

    #set up model with sequential
    recurModel = Sequential()
    #use a timeDistributed(conv2D) followed by timeDistributed(maxpooling2d) followed by timeDistributed(flatten())
    recurModel.add(TimeDistributed(Conv2D(neuronLayer[0], kernel_size=(kernel, kernel), strides=(strideC, strideC),activation='relu', input_shape=train_data.shape[1:])))
    recurModel.add(TimeDistributed(MaxPooling2D(pool_size=(pool, pool), strides=(strideP, strideP))))
    recurModel.add(TimeDistributed(Flatten()))

    # the above scheme should allow for the images to be processed similar to movie frames.
    #EMBEDDING LAYER? LOOK into; seems like its only for language processing?

    # use if statement to determine if the user wants to use LSTM or RNN
        #if LSTM then set up the LSTM network
    for layer in range(len(neuronLayer)):
        if boolLSTM:
            recurModel.add(LSTM(neuronLayer[layer], return_sequences = True))
        else:
            print("RNN?")
    if drop > 0:
            recurModel.add(Dropout(drop))

    recurModel.add(Dense(2, activation = 'softmax'))
    recurModel.summary()

    #compile and train the model
    #recurModel.compile()
    #recur_hist = convModel.fit()

    #plot stuff
    #makePlots(recur_hist, outputFile)

    return recurModel

#RBFN
    #do stuff, look at what might be a good starting point
def rbfn():
    '''
    implements a radial bayes neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        train_data : data to train on
        train_label : labels of training data

    Returns:
        rbfModel : a trained keras RBF network
    '''
    print("radial bayes neural network")

#siamese nn
    # DO NOT ATTEMPT UNTIL LAST. can use example from https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d
def snn():
    '''
    implements a siamese neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        train_data : data to train on
        train_label : labels of training data

    Returns:
        siameseModel : a trained keras siamese network
    '''
    print("siamese neural network")

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

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d-%H_")
    print(date)

    #for all dataset directories & all data in each: need to walk through the various dierctories that contain each dataset
    #can change this directory once run location on cheyenne is selected

    for folder in os.listdir('.'):

        #extract the data from all the necessary files for the given lead time
        # need to change
        if not '.' in folder:
            print('folder', folder)
            #get lead time to save to output file name
            lead,extra = folder.split("_")
            outputDL = date + lead + "_"
            print(outputDL)
            outputFile = outputDir + outputDL

            for f in os.listdir(folder):

                if not f.startswith('.'):

                    with open(folder + "/" + f + "/" + f + ".txt", 'rb') as file:
                        if f == 'X_train':
                            train_data = np.array(pickle.load(file))
                        if f == 'X_dev':
                            dev_data = pickle.load(file)
                        if f == 'X_val':
                            test_data = pickle.load(file)
                        if f == 'Y_train':
                            train_label = pickle.load(file)
                        if f == 'Y_dev':
                            dev_label = pickle.load(file)
                        if f == 'Y_val':
                            test_label = pickle.load(file)

    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    #print(train_data[0])
    np.asarray(train_data)
    print(train_data[0].shape)
    print(train_data[15].shape)

    denseNN = dnn([16,16,2], 0.5, 0.0001, 0.99, 1e-4, True, 1,train_data, train_label,dev_data, dev_label, outputFile) #these are all negins values right now.
    #convNN = cnn([20,50,500,2], [5,5], [2,2], [1,1], [2,2], 0.01, 1000, train_data, train_label, dev_data, dev_label, outputDL) # these are the lenet values
    #recurrNN = rnn()
    #radialBayesNN = rbfn()
    #siameseNN = snn()

    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
