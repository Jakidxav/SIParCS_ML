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
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
import sklearn.metrics
import pickle
import os
import datetime


#ROC calculations. will need to use this V datasets on all algorithms
def calculateAUROC(fp, tp):
    auroc = 0
    '''
    implements an AUROC calculation. also graphs the ROC curve

    Arguments:
        tp : array containing true positives for each run
        fp : array containing false positives for each run
        tp and fp should be the same length! values must be 0< & <1

    Returns:
        auroc : double value
    '''
    plt.plot(fp, tp, "b")
    plt.plot([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], "r--") # this is the .5 line, just for reference
    plt.savefig('test.png') # will need to change naming scheme/setup once algorithms and various datasets are implemented.
    #will have to modify name based off of set, algorithm, and hyperparameters. will also need to change save location
    return sklearn.metrics.auc(fp, tp, reorder = False)

#brier score and brier skill score. both methods written by Negin
def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    return 1.0 - brier_score_keras(obs, preds) / climo


#dense nn
    # based off of dnn from Negin. just need to focus on optimizing
def dnn(neuronLayer, drop, learnRate, momentum, decay,boolNest, iterations, train_data, train_label, dev_data, dev_label):
    '''
    implements a dense neural network

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

    Returns:
        denseModel : a trained keras dense network

    Example :
        dnn([16,16,1], 0.5, 0.0001, 0.99, 1e-4, True, train_data, train_label)
    '''
    print("dense neural network")
    #initilaize model with Sequential()
    denseModel = Sequential()
    #add first layers
    denseModel.add(AveragePooling2D(poolesize = (32,32))(train_data.shape)) # negin used this as the first layer. need to double check syntax
    denseModel.add(Flatten())

    for layer in neuronLayer:
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dropout(drop))
        denseModel.add(Dense(neuronLayer[layer], activation = 'relu'))

        #this is the output layer; # neurons should be equal to 1
        if(layer == (len(neuronLayer) - 1)):
            denseModel.add(Dropout(drop))
            denseModel.add(Dense(neuronLayer[layer], activation = 'sigmoid'))

    #define optimizer
    opt_dense = SGD(lr=learnRate, momentum= momentum, decay= decay, nesterov= boolNest)
    denseModel.summary()

    #compile
    denseModel.compile(opt_dense, "mse", metrics=[brier_skill_score_keras])

    dense_hist = denseModel.fit(train_data, train_label, batch_size=256, epochs=iterations, verbose=2,validation_data=(dev_data, dev_label))
    #predict
    #denseModel.predict(test_data)

    #evaluate
    #score = dense_model.evaluate(test_data, test_label, verbose=1)
    #print(score)

    return denseModel

#cnn
    # start with 3x convlayer and poolLayer repeats.
    #WILL NEED TO CHANGE THE KERNEL, POOL, STRIDE PARAMS TO BE LISTS SO THAT ALEXNETS/OTHER NETS
    #WITH VARYING STRIDES ETC CAN BE IMPLEMENTED
def cnn(neuronLayer, kernel, pool,strideC, strideP, learnRate, iterations, train_data, train_label, dev_data, dev_label):
    '''
    implements a convolutional neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        kernel : size of conv kernel
        pool : pool_size amount
        strideC : length of conv stride
        strideP : length of pool stride
        learnRate : learning rate
        iterations : number of iterations to train the model
        train_data : data to train on (numpy array)
        train_label : labels of training data (numpy array)

    Returns:
        convModel : a trained keras convolutional network

    Example:
        cnn([32,64, 1000, 1], 5, 2, 1, 1, 0.01, 1000, train_data, train_label, dev_data, dev_label)
        lenet would be: cnn([20,50,500,2], 5,2,1,2, 0.01, 1000, train_data, train_label, dev_data, dev_label)
    '''
    print("convoultional neural network")

    #initilaize model with Sequential
    convModel = Sequential()

    #add first conv and pooling layers
    convModel.add(Conv2D(neuronLayer[0], kernel_size=(kernel, kernel), strides=(strideC, strideC),activation='relu', input_shape=train_data.shape))
    convModel.add(MaxPooling2D(pool_size=(pool, pool), strides=(strideP, strideP)))

    for layer in range(1, len(neuronLayer) - 3):
        convModel.add(Conv2D(neuronLayer[layer], kernel_size = (kernel,kernel), activation='relu'))
        convModel.add(MaxPooling2D(pool_size=(pool, pool), strides=(strideP, strideP)))

    convModel.add(Flatten())
    convModel.add(Dense(neuronLayer[len(neuronLayer) - 2], activation='relu'))
    convModel.add(Dense(neuronLayer[len(neuronLayer) - 1], activation='softmax'))

    convModel.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=learnRate),metrics=[brier_skill_score_keras])
    convModel.fit(train_data, train_label,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_data, test_label))

    return convModel
# rnn
    # do stuff. look at what might be a good starting point; could try LSTM??
def rnn():
    '''
    implements a recurrent neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        boolLSTM : boolean representing to use LSTM net or simple RNN
        train_data : data to train on
        train_label : labels of training data
        dev_data : data for dev set/validation
        dev_label : labels for the dev set

    Returns:
        recurrentModel : a trained keras recurrent network
    '''
    print("recurrent neural network")

    #set up model with sequential
    #use a timeDistributed(conv2D) followed by timeDistributed(maxpooling2d) followed by timeDistributed(flatten())
    # the above scheme should allow for the images to be processed similarly to a movie prediction.
    #EMBEDDING LAYER? LOOK into

    # use if statement to determine if the user wants to use LSTM or RNN

    #if LSTM then set up the LSTM network

    # for layers in neuronLayer
        #add LSTM layer
        #if using Dropout
            #add dropout layer

    #add a time distrubeted layer with dense
    #add final activationlayer

    #else set up the rnn
        #

    #compile the model
    # 

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
    X_train_filename = 'X_train_scratch.txt'

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d_")
    print(date)

    #for all dataset directories & all data in each: need to walk through the various dierctories that contain each dataset
    #can change this directory once run location on cheyenne is selected
    '''
    for folder in os.listdir('/glade/work/joshuadr/IPython'):

        #get lead time to save to output file name
        lead,extra = folder.split("_")
        outputPrefix = date + lead + "_"
        print(outputPrefix)

        #extract the data from all the necessary files for the given lead time
        # need to change
        with open(folder + "/X_train/X_train.txt", 'rb') as file:
            train_data = pickle.load(file)

        with open(folder + "/X_dev/X_dev.txt", 'rb') as file:
            dev_data = pickle.load(file)

        with open(folder + "/X_val/X_val.txt", 'rb') as file:
            test_data = pickle.load(file)

        with open(folder + "/Y_train/Y_train.txt", 'rb') as file:
            train_label = pickle.load(file)

        with open(folder + "/Y_dev/Y_dev.txt", 'rb') as file:
            dev_label = pickle.load(file)

        with open(folder + "/Y_val/Y_val.txt", 'rb') as file:
            test_label = pickle.load(file)

        #train all nteworks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?

        #dnn([16,16,1], 0.5, 0.0001, 0.99, 1e-4, True, 150,train_data, train_label,dev_data, dev_label) #these are all negins values right now.
        #cnn([20,50,500,2], 5,2,1,2, 0.01, 1000, train_data, train_label, dev_data, dev_label) # these are the lenet values
        #rnn()
        #rbfn()
        #snn()

        #run test sets.
            # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
    '''
    print(calculateAUROC([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], [.2,.5,.9,.9,.9,.9,.85,.85,.9,1.0]))
