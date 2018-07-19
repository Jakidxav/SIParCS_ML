'''
parallel 2x NN for using soil etc data with the SST data. methods are different in the sense that each nn method returns an uncompiled
nn for one half of the parallel net. each net should be returned for the given dataset then merged for the final decision.
after the merge a model summary, compiling, and fitting should occur

need to read in 2 different datasets
'''
#imports
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from keras import layers, models
from keras.layers.merge import average
from keras.models import Sequential, Model, save_model, load_model
import keras.backend as K
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, TimeDistributed, LSTM, Dropout, BatchNormalization, Average
from keras.metrics import binary_accuracy, mean_squared_error
from keras.losses import  binary_crossentropy
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
import sklearn.metrics as skm
from tensorflow import metrics as tf
import h5py
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
outputDir = "./data/"

#example for generating list of random numbers for grid search
# list = random.sample(range(min, max), numberToGenerate)

# opt_merge = Adam(lr=learningRate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
# merged_model.fit(train_data, train_label,batch_size=128,epochs=epochs,verbose=2,validation_data=(dev_data, dev_label), class_weight = {0:0.5, 1:1})

#hyperparameters and paramters

epochs = 219

#parameters for Adam optimizaiton
learningRate = 0.01
decay = 1e-4
beta_1=0.9
beta_2=0.999
epsilon=None
amsgrad=False

# create tab delim text file of roc stuff.
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

#make plots
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

#write file
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


#main
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
    #netType = sys.argv[1] #can be dense, conv, merged, alex, or all
    #stationNum = sys.argv[2] # the station to be used. should be in the form of station1
    #print(netType)

    #start setting up the name for the output file
    date = datetime.datetime.now().strftime("%y%m%d_")
    #print(date)

    outputDL = date + "30_"
    #print(outputDL)
    outputFile = outputDir + outputDL +'_merged'

    '''
    X_train_filename = '/glade/work/joshuadr/IPython/30_lead/X_train/X_train.txt'
    X_dev_filename = '/glade/work/joshuadr/IPython/30_lead/X_dev/X_dev.txt'
    X_val_filename = '/glade/work/joshuadr/IPython/30_lead/X_val/X_val.txt'

    Y_train_filename = '/glade/work/joshuadr/IPython/30_lead/Y_train/station1/Y_train.txt'
    Y_dev_filename = '/glade/work/joshuadr/IPython/30_lead/Y_dev/station1/Y_dev.txt'
    Y_val_filename = '/glade/work/joshuadr/IPython/30_lead/Y_val/station1/Y_val.txt'
    '''
    #will need to adjust the following file input once another dataset is created. 
    X_train_filename = './30_lead/X_train/X_train.txt'
    X_dev_filename = './30_lead/X_dev/X_dev.txt'
    X_val_filename = './30_lead/X_val/X_val.txt'

    Y_train_filename = './30_lead/Y_train//Y_train.txt'
    Y_dev_filename = './30_lead/Y_dev/Y_dev.txt'
    Y_val_filename = './30_lead/Y_val/Y_val.txt'

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

    #open up nonsst data

    #reshape all data files.
    train_data2 = train_data.reshape(-1,120,340, 1)
    dev_data2 = dev_data.reshape(-1,120,340,1)
    test_data2 = test_data.reshape(-1,120,340,1)

    #load pretrained models. learn weights just for deciding which nn should play a bigger role in determining final output
    #still need a merge layer for the NN outputs

    start = time.time()

    #load in and test first model
    seqmodel = load_model("./data/Recur/lrE2/180716__30_0_rnn.h5")
    score = seqmodel.evaluate(train_data2, train_label, verbose=1)
    print("%s: %.2f%%" % (seqmodel.metrics_names[1], score[1]*100))

    #convert first model to functional
    input_layer = layers.Input(batch_shape=seqmodel.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seqmodel.layers:
        prev_layer = layer(prev_layer)
        print(type(prev_layer))

    funcmodel = models.Model([input_layer], [prev_layer])
    funcmodel.summary()


    #load in and test second model
    seqmodel2 = load_model("./data/Dense/lrE2/180716__30_0_dnn.h5")
    score = seqmodel2.evaluate(train_data2, train_label, verbose=1)
    print("%s: %.2f%%" % (seqmodel2.metrics_names[1], score[1]*100))

    #convert second model to functional
    input_layer2 = layers.Input(batch_shape=seqmodel2.layers[0].input_shape)
    prev_layer2 = input_layer2
    for layer in seqmodel2.layers:
        prev_layer2 = layer(prev_layer2)

    funcmodel2 = models.Model([input_layer2], [prev_layer2])
    funcmodel2.summary()

    #rename layers in the second model if they exist in first model. if no renaming occurs an error will occur upon merging :(
    #check to see if the name of prev_layer2 exists in funcmodel
    #this should only have to be done once for each layer type cause the layer increments should be adjusted accordingly.
    print("layer checking \n")
    for layer2 in funcmodel2.layers:
        #if it does, find all layers in funcmodel of that layer type. increment the largest number by 1 and save
        for layer in funcmodel.layers:
            if layer2.name == layer.name:
                #check through all prefix_num until no more wiht prefix exist in funcmodel. change layer2.name to prefix_maxNum+1
                lType, num = layer.name.split('_')
                num = int(num)
                for l in funcmodel.layers:

                    lt, n = l.name.rsplit('_',1)
                    n = int(n)
                    #if more than one layer of lType exists, then save the number to num
                    if lt == lType:
                        num = n

                # change the name so the number at the end of the name is now the saved number
                layer2.name = lType + '_' + str(num)
                #then check through funcmodel2 to increment each layer with prefix by 1
                for l2 in funcmodel2.layers:

                    lt2, n2 = l2.name.rsplit('_',1)
                    n2 = int(n2)
                    if lt2 == lType:
                        l2.name = lt2 +  '_' + str(n2 + 1)

    #merge models
    merged = average([prev_layer, prev_layer2])

    #need to get number of dense layers to name final dense layer correctly.
    dense = 1

    for layer in funcmodel.layers:
        lType, num = layer.name.rsplit('_',1)
        if lType == 'dense':
            dense = dense + 1

    for layer in funcmodel2.layers:
        lType, num = layer.name.rsplit('_',1)
        if lType == 'dense':
            dense = dense + 1

    #create output dense layer
    final = Dense(1, activation='sigmoid', name='dense' +  '_' + str(dense + 1))(merged)

    merged_model = Model(input = [input_layer, input_layer2], output = final)

    merged_model.summary()

    opt_merge = Adam(lr=learningRate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

    merged_model.compile(loss=mean_squared_error,optimizer=opt_merge,metrics=[binary_accuracy])
    merged_hist = merged_model.fit(train_data, train_label,batch_size=128,epochs=epochs,verbose=2,validation_data=(dev_data, dev_label), class_weight = {0:0.5, 1:1})

    #calculate ROC info
    train_pred = merged_model.predict(train_data).ravel()
    dev_pred = merged_model.predict(dev_data).ravel()
    fpr_train, tpr_train, thresholds_train = skm.roc_curve(train_label,train_pred)
    fpr_dev, tpr_dev, thresholds_dev = skm.roc_curve(dev_label, dev_pred)

    #save roc info to a tab delim file
    rfile = open(outputFile + '_roc_vals.txt', "w+")
    rocFile(rfile, fpr_train, tpr_train, fpr_dev, tpr_dev)
    #generate plts and save
    makePlots(merged_hist, outputFile, "Merged Neural Net", fpr_train, tpr_train, fpr_dev, tpr_dev)

    #save the model
    merged_model.save(outputFile+ '.h5')

    print("runtime ",time.time() - start, " seconds")
