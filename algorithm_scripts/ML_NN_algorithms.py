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
    will probably want to automate this. EACH NN METHOD SHOULD TAKE IN # NEURONS
    AS AN ARRAY WHERE EACH INDEX IS A NONINPUT LAYER, EACH RELEVANT TUNING PARAMETER. should use a for loops when setting up
    layers to go through number of neurons and layers. should make each method as versitile as possible s.t. network optimization
    can be easily optimized.

benefits of embedding layers? need to use?

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
def dnn(neuronLayer, iterations, train_data, train_label):
    '''
    implements a dense neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        train_data : numpy array data to train on
        train_label : numpy array labels of training data

    Returns:
        denseModel : a trained keras dense network
    '''
    print("dense neural network")
    #initilaize model with Sequential()
    denseModel = Sequential()

    denseModel.add(Dense(neuronLayer[0], train_data.shape, activation = 'relu'))

    for i in range(1,len(neuronLayer)):
        #add layers to denseModel with # of neurons at neuronLayer[i] and apply dropout
        denseModel.add(Dense(neuronLayer[i], activation = 'relu'))
        denseModel.add(Dropout(0.5))

        if(i == (len(neuronLayer) - 1):
            denseModel.add(Dense(neuronLayer[i], activation = 'sigmoid'))

    #define optimizer
    opt_dense = SGD(lr=0.0001, momentum=0.99, decay=1e-4, nesterov=True)

    #compile
    denseModel.compile(opt_dense, "mse", metrics=[brier_skill_score_keras])
    #.fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    #.evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
    #look at performance; requires a predict,

    return denseModel

#cnn
    # start with 3x convlayer and poolLayer repeats.
    #activation functions to start with: relu or LeakyReLU
def cnn():
    '''
    implements a convolutional neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        train_data : data to train on
        train_label : labels of training data

    Returns:
        convModel : a trained keras convolutional network
    '''
    print("convoultional neural network")

# rnn
    # do stuff. look at what might be a good starting point; could try LSTM??
def rnn():
    '''
    implements a recurrent neural network

    Arguments:
        neuronLayer : array containing the number of neurons perlayer excluding input layer
        remaining tuning parameters
        iterations : number of iterations to train the model
        train_data : data to train on
        train_label : labels of training data

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

    #for all dataset directories & all data in each:
        #train all nteworks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
        #dnn([],)
        #run AUROC calculation on each trained models

        #run dev sets and AUROC calculation

        #run test sets.
            # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)

    print(calculateAUROC([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], [.2,.5,.5,.6,.7,.8,.85,.85,.9,1.0]))
