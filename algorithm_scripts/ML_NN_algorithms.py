"""

@author Negin Sobhani, Joshua Driscol, Karen Stengel

this script is designed to use keras with Tensorflow as the backendself.
it should be used in conjunction with ____.py which should set up/create necessary
datasets from the McKinnon et al. dataself.

this script will take the premade datasets and run them through the various ML NN algorithms
and return ROC scores; can use the binary_accuaracy method in keras as a starting pointself.
This script will also include optimizers so that we can get the best
scores possible.

NEED TO FIGURE OUT HOW TO DO THE FOLLOWING IN KERAS EFFECIENTLY
algorithms to include:
    dense nn - already created by Negin so will just need to try optimizing
    CNN - also already tried by Negin so will need to try optimizing.
    siamese nn - possibly use? need to look into best implementation...
    RNN - not yet tried. will need to set up and OPTIMIZE
    RBFN - ? maybe use? could allow for an interesing analysis....

will need each algorithm method to take in certain parameters to make optimizing easier.
    keras optimizers are specifically for things like momentum, learning rate, epislon, etc etc
    thus, we will have to manually (or write a method to) change number on neurons and/or layers (and # in each layer)
    will probably want to automate this. EACH NN METHOD SHOULD TAKE IN # NEURONS
    AS AN ARRAY WHERE EACH INDEX IS A NONINPUT LAYER, EACH RELEVANT TUNING PARAMETER

datasets:
    should be set up with various lead prediction times. ex: 10, 30, 45, 60 days in advance
    should also use a control set with randomly shuffled labels
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import sklearn.metrics


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
    plt.plot([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], "r--") # this is the .5 line
    plt.savefig('test.png') # will need to change naming scheme/setup once algorithms and various datasets are implemented.
    #will have to modify name based off of set, algorithm, and hyperparameters. will also need to change save location
    return sklearn.metrics.auc(fp, tp, reorder = False)

#dense nn
    # most of this should be from Negin. just need to focus on optimizing
def dnn():
    '''
    implements a dense neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model

    Returns:
        denseModel : a trained keras dense network
    '''
    print("dense neural network")

#cnn
    # start with 3x convlayer and poolLayer repeats.
    #activation functions to start with:
def cnn():
    '''
    implements a convolutional neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model

    Returns:
        convModel : a trained keras convolutional network
    '''
    print("convoultional neural network")

# rnn
    # do stuff. look at what might be a good starting point
def rnn():
    '''
    implements a recurrent neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model

    Returns:
        recurrentModel : a trained keras recurrent network
    '''
    print("recurrent neural network")

#RBFN
    #do stuff, look at what might be a good starting point
def rbfn():
    '''
    implements a radial bayes neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model

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

    Returns:
        siameseModel : a trained keras siamese network
    '''
    print("siamese neural network")

#main stuff
    #this should read in each dataset and call the NN algorithms.
if __name__ == "__main__":
    print("you are in main.")

    #for all dataset directories & all data in each:
        #train all nteworks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
        #run AUROC calculation on each trained models

        #run dev sets and AUROC calculation

        #run test sets.
    print(calculateAUROC([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], [.2,.5,.5,.6,.7,.8,.85,.85,.9,1.0]))
