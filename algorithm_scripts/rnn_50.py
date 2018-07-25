
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

#will allow for files to
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#path for data output. each file should contain all used params after training and metrics
outputDir = "./data/Recur/weights/"

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)
posWeight = 1
trials = 5
#hyperparameters and paramters
#SGD parameters
dropout = 0.5
momentum = 0.99

learningRate = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
epochs = 300
decay = 1e-4
batch = 128


boolNest = True

#parameters for conv/pooling layers
strideC = [5,5, 1]
strideP = [2,2]
kernel = [5, 5,1]
pool = [2,2]

#parameters for Adam optimizaiton
boolAdam = False #change to false if SGD is desired

beta_1= 0.9
beta_2= 0.999

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

    #training confusion matrix
    train_class = np.round(train_pred)
    for x in train_class:
        int(x)

    cm_train = skm.confusion_matrix(train_label, train_class)

    xlabel = 'Predicted labels'
    ylabel = 'True labels'
    train_title = 'Training ROC Confusion Matrix'


    ax = sns.heatmap(cm_train, annot=True, fmt='d', cbar=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(train_title)
    ax.figure.savefig(output + '_train_cm.pdf', format='pdf')
    plt.cla()
    ax.clear()

    #dev confusion matrix
    dev_class = np.round(dev_pred)
    for y in dev_class:
        int(y)

    cm_dev = skm.confusion_matrix(dev_label, dev_class)

    dev_title = 'Dev Set ROC Confusion Matrix'

    ax2 = sns.heatmap(cm_dev, annot=True, fmt='d', cbar=False)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(dev_title)
    ax2.figure.savefig(output + '_dev_cm.pdf', format='pdf')
    plt.cla()
    ax2.clear()

def writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight):
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


# rnn
    # do stuff. look at what might be a good starting point; could try LSTM??
def rnn(neuronLayer, kernel, pool, strideC, strideP, drop, learnRate, momentum, decay,boolNest,boolLSTM, boolAdam, b1, b2, epsilon, amsgrad, iterations, train_data, train_label, dev_data, dev_label, outputDL, searchNum, batch, posWeight):
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
    writeFile(file,neuronLayer, iterations, boolLSTM, boolAdam, boolNest, drop, kernel, pool, strideC, strideP, momentum, decay, learnRate, b1, b2, epsilon, amsgrad, searchNum, posWeight)

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
    recur_hist = recurModel.fit(train_data, train_label,batch_size=batch,epochs=iterations,verbose=2,validation_data=(dev_data, dev_label), class_weight = {0:1, 1:posWeight})

    #calculate ROC info
    train_pred = recurModel.predict(train_data).ravel()
    dev_pred = recurModel.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    rfile = open(outputFile + '_roc_vals.txt', "w+")
    #rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    makePlots(recur_hist, outputFile, "LSTM Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev, train_pred, dev_pred)

    #recurModel.save(outputFile+ '.h5')

    return recurModel, skm.auc(fpr_dev,tpr_dev), train_pred, dev_pred, thresholds_train, thresholds_dev, tpr_train, fpr_train, tpr_dev, fpr_dev



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

    outputDL = date + "_50_"
    #print(outputDL)
    outputFile = outputDir + outputDL

    X_train_filename = '/glade/work/joshuadr/IPython/X/50_lead/X_train/X_train.txt'
    X_dev_filename = '/glade/work/joshuadr/IPython/X/50_lead/X_dev/X_dev.txt'
    X_val_filename = '/glade/work/joshuadr/IPython/X/50_lead/X_val/X_val.txt'

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
    bestRnnAUROC = 0
    bestRnnSearchNum = 0
    bestTrial = 0


    #train all networks. call each NN method with corresponding parameters. manually change to tune or can set up an automation?
    #each method will finish adding to the output file name and write all hyperparameters/parameters and metrics info to below file.
    start = time.time()
    i = 0
    for lr in learningRate:

        for t in range(trials):
            outputSearch = outputFile + str(i) + "." +  str(t) +"_"

            recurrNN, rnnAUROC, train_pred, dev_pred, train_thresh, dev_thresh, tpr_train, fpr_train, tpr_dev, fpr_dev = rnn([20,60],kernel, pool, strideC, strideP, dropout, lr, momentum, 1.0e-4, boolNest,True, boolAdam,beta_1, beta_2, epsilon, amsgrad, epochs, train_data2, train_label, dev_data2, dev_label,outputSearch, i, batch, posWeight)

            recurrNN.save(outputSearch +'.h5')

            train_pred_filename = outputSearch+'train_pred_.txt'
            with open(train_pred_filename, 'wb') as f:
                pickle.dump(train_pred, f)

            dev_pred_filename = outputSearch+'dev_pred_.txt'
            with open(dev_pred_filename, 'wb') as g:
                pickle.dump(dev_pred, g)

            train_thresh_filename = outputSearch+'train_thresh_.txt'
            with open(train_thresh_filename, 'wb') as h:
                pickle.dump(train_thresh, h)

            dev_thresh_filename = outputSearch+'dev_thresh_.txt'
            with open(dev_thresh_filename, 'wb') as k:
                pickle.dump(dev_thresh, k)

            tpr_train_filename = outputSearch+'tpr_train_.txt'
            with open(tpr_train_filename, 'wb') as h:
                pickle.dump(tpr_train, h)

            fpr_train_filename = outputSearch+'fpr_train_.txt'
            with open(fpr_train_filename, 'wb') as h:
                pickle.dump(fpr_train, h)

            tpr_dev_filename = outputSearch+'tpr_dev_.txt'
            with open(tpr_dev_filename, 'wb') as h:
                pickle.dump(tpr_dev, h)

            fpr_dev_filename = outputSearch+'fpr_dev_.txt'
            with open(fpr_dev_filename, 'wb') as h:
                pickle.dump(fpr_dev, h)

        if rnnAUROC > bestRnnAUROC:
            bestRnnAUROC = rnnAUROC
            bestRnnParams = [lr]
            bestRnnSearchNum = i
            bestTrial = t

        i += 1


    bfile.write("best RNN AUROC for dev set: " + str(bestRnnAUROC) + "\n")
    bfile.write("best RNN search iteration for dev set: " + str(bestRnnSearchNum) + "\n")
    bfile.write("best RNN search iteration for dev set: " + str(bestTrial) + "\n")


    print("runtime ",time.time() - start, " seconds")
    #run test sets.
    # ex model.predict(self, x, batch_size=None, verbose=0, steps=None)
