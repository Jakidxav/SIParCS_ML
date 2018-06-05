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

will need each algorithm method to take in certain parameters to make optimizing easierself.
will need to check for best way to do this using either keras optimizers or self made optimization techniques

datasets:
    should be set up with various lead prediction times. ex: 10, 30, 45, 60 days in advance
    should also use a control set with randomly shuffled labels
"""

#ROC calculations. will need to use this V datasets on all algorithms

#main stuff
    #this should read in each dataset and call the NN algorithms.

#dense nn
    # most of this should be from Negin. just need to focus on optimizing

#cnn
    # start with 3x convlayer and poolLayer repeats.
    #activation functions to start with:

# rnn
    # do stuff. look at what might be a good starting point

#RBFN
    #do stuff, look at what might be a good starting point

#siamese nn
    # DO NOT ATTEMPT UNTIL LAST. can use example from https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d
