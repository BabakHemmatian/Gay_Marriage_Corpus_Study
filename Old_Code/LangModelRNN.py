#!/python27
# -*- coding: utf-8 -*-
### Tuning ###
### import the required modules and functions
from __future__ import print_function
from collections import defaultdict
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
from random import sample
from math import floor,ceil
### Set hyperparameters
epochs = 3 # number of times the model goes throught the training set to update weights
MaxVocab = 100000 # maximum size of the vocabulary
batchSz = 50 # number of parallel batches
windowSz = 20 # window size
embedSz = 128 # embedding size
hiddenSz = 512 # number of units in the recurrent layer
ff1Sz = 1000 # number of units in the first feedforward layer
ff2Sz = 1000 # number of units in the second feedforward layer
keepP = 0.5 # 1 - dropout rate
print("Embedding size = " + str(embedSz))
print("Recurrent layer size = " + str(hiddenSz))
print("1st feedforward layer size = " + str(ff1Sz))
print("2nd feedforward layer size = " + str(ff2Sz))
# set default file encoding
reload(sys)
sys.setdefaultencoding('utf8')
### Prepare data ###
### Load the number of comments and divide them into training, development and test sets (70,15,15 percents)
# timedict = {}
# with open("RC_Count_Dict.txt") as f:
#     for line in f:
#         (key, val) = line.split()
#         timedict[key] = int(val)
timelist = []
with open("RC_Count_List.txt",'r') as f:
    for line in f:
        timelist.append(int(line))
num_comm = timelist[-1] # number of comments
num_train = int(ceil(0.7*num_comm)) # size of training set
train_set = sample(range(num_comm),num_train) # choose training comments at random
remaining = [x for x in range(num_comm) if x not in train_set]
num_dev = int(floor(len(remaining)/2)) # size of development set
dev_set = sample(remaining,num_dev) # choose development comments at random
test_set = [x for x in remaining if x not in dev_set] # use the rest as test set
# write the sets to file
np.savetxt("train_set",train_set)
np.savetxt("dev_set",dev_set)
np.savetxt("test_set",test_set)
### Build the vocabulary
## initialize the vocabulary with various UNKs
V = {"*STOP*":0,"*STOP2*":1,"*UNK*":2,"*UNKED*":3,"*UNKS*":4,"*UNKING*":5,"*UNKLY*":6,"*UNKER*":7,"*UNKION*":8,"*UNKAL*":9,"*UNKOUS*":10}
# timer
print("Started parsing the training set at " + time.strftime('%l:%M%p'))
## record word frequency
frequency = defaultdict(int)
fin = open('nn_prep.txt','r')
for comment in fin: # for each comment
    for token in comment.split(): # for each word
        frequency[token] += 1 # count the number of occurrences
## read the training set data and create the vocabulary
fin.seek(0)
indexed_train = []
for counter,comm in enumerate(fin): # for each comment
    if counter in train_set: # if it belongs in the training set
        for word in comm.split(): # for each word
            if frequency[word] > 5: # filter non-frequent words
                if word in V.keys(): # if the word is already in the vocabulary
                    indexed_train.append(V[word]) # record the relevant index
                else: # if the word is not in vocabulary
                    if len(V)-11 <= MaxVocab: # if the vocabulary still has room (not counting STOPs and UNKs)
                        V[word] = len(V) # give it an index
                        indexed_train.append(V[word]) # append it to the list of words
                    else: # if the vocabulary doesn't have room, assign the word to an UNK according to its suffix or lack thereof
                        if word.endswith("ed"):
                            indexed_train.append(3)
                        elif word.endswith("s"):
                            indexed_train.append(4)
                        elif word.endswith("ing"):
                            indexed_train.append(5)
                        elif word.endswith("ly"):
                            indexed_train.append(6)
                        elif word.endswith("er"):
                            indexed_train.append(7)
                        elif word.endswith("ion"):
                            indexed_train.append(8)
                        elif word.endswith("al"):
                            indexed_train.append(9)
                        elif word.endswith("ous"):
                            indexed_train.append(10)
                        else: # if the word doesn't have any easily identifiable suffix
                            indexed_train.append(2)
# save the vocabulary to file
vocab = open("dict.txt",'a+')
for word,index in V.iteritems():
    print(word+" "+str(index),end='\n',file=vocab)
vocab.close
# save the indices for training set to file
np.savetxt("indexed_train", indexed_train)
# timer
print("Finished parsing the training set at " + time.strftime('%l:%M%p'))
### read the development and test sets and assign the relevant indices
fin.seek(0)
indexed_dev = []
indexed_test = []
for counter,comm in enumerate(fin): # for each comment
    if counter in dev_set: # if it belongs in the development set
        for word in comm.split():
            if frequency[word] > 5: # filter non-frequent words
                if word in V.keys():
                    indexed_dev.append(V[word])
                else:
                    if word.endswith("ed"):
                        indexed_dev.append(3)
                    elif word.endswith("s"):
                        indexed_dev.append(4)
                    elif word.endswith("ing"):
                        indexed_dev.append(5)
                    elif word.endswith("ly"):
                        indexed_dev.append(6)
                    elif word.endswith("er"):
                        indexed_dev.append(7)
                    elif word.endswith("ion"):
                        indexed_dev.append(8)
                    elif word.endswith("al"):
                        indexed_dev.append(9)
                    elif word.endswith("ous"):
                        indexed_dev.append(10)
                    else:
                        indexed_dev.append(2)
    elif counter in test_set: # if the comment belongs in the test set
        for word in comm.split():
            if frequency[word] > 5:
                if word in V.keys():
                    indexed_test.append(V[word])
                else:
                    if word.endswith("ed"):
                        indexed_test.append(3)
                    elif word.endswith("s"):
                        indexed_test.append(4)
                    elif word.endswith("ing"):
                        indexed_test.append(5)
                    elif word.endswith("ly"):
                        indexed_test.append(6)
                    elif word.endswith("er"):
                        indexed_test.append(7)
                    elif word.endswith("ion"):
                        indexed_test.append(8)
                    elif word.endswith("al"):
                        indexed_test.append(9)
                    elif word.endswith("ous"):
                        indexed_test.append(10)
                    else:
                        indexed_test.append(2)
# timer
print("Finished parsing the development and test sets at " + time.strftime('%l:%M%p'))
# close the data file
fin.close()
# save the indices for development and test sets to file
np.savetxt("indexed_dev", indexed_dev)
np.savetxt("indexed_test", indexed_test)
### set up the computation graph ###
### create placeholders for input, output
inpt = tf.placeholder(tf.int32, shape=[None,None])
answr = tf.placeholder(tf.int32, shape=[None,None])
### set up the variables
# initial embeddings
E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))
# look up the embeddings
embed = tf.nn.embedding_lookup(E, inpt)
# define the recurrent layer (Gated Recurrent Unit)
rnn= tf.contrib.rnn.GRUCell(hiddenSz)
initialState = rnn.zero_state(batchSz, tf.float32)
output, nextState = tf.nn.dynamic_rnn(rnn, embed,initial_state=initialState)
# create weights and biases for three feedforward layers
W1 = tf.Variable(tf.random_normal([hiddenSz,ff1Sz], stddev=0.1))
b1 = tf.Variable(tf.random_normal([ff1Sz], stddev=0.1))
l1logits = tf.nn.relu(tf.tensordot(output,W1,[[2],[0]])+b1)
l1Output = tf.nn.dropout(l1logits,keepP) # apply dropout
W2 = tf.Variable(tf.random_normal([ff1Sz,ff2Sz], stddev=0.1))
b2 = tf.Variable(tf.random_normal([ff2Sz], stddev=0.1))
l2Output = tf.nn.relu(tf.tensordot(l1Output,W2,[[2],[0]])+b2)
W3 = tf.Variable(tf.random_normal([ff2Sz,len(V)], stddev=0.1))
b3 = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
### calculate loss
# calculate logits
logits = tf.tensordot(l2Output,W3,[[2],[0]])+b3
# calculate sequence cross-entropy loss
xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=answr,weights=tf.ones([batchSz,windowSz],tf.float32))
loss = tf.reduce_mean(xEnt)
### training with AdamOptimizer
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
### training the network ###
### create the session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
state = sess.run(initialState)
# number of windows and length of each batch for the training set
batchLen = int(floor(float((len(indexed_train)))/float(batchSz)))
number_of_windows = int(floor(float(batchLen)/float(windowSz)))
# number of windows and batch length for the development set
batchLen = int(floor(float((len(indexed_dev)))/float(batchSz)))
number_of_windows = int(floor(float(batchLen)/float(windowSz)))
# arrange the input in batches
new_input = np.asarray(indexed_train[0:batchSz*batchLen])
new_input = np.reshape(new_input,(batchSz,batchLen))
train_perplexity = np.empty(epochs) # initialize vector to store training set preplexity
dev_perplexity = np.empty(epochs) # initialize vector to store development set perplexity
### train the network
for k in range(epochs): # for each epoch
    ## train on the training set
    Loss = 0 # reset the loss
    for i in range(number_of_windows): # for each window
        # initialize vectors for feeding data and desired output
        inputs = np.zeros([batchSz,windowSz])
        answers = np.zeros([batchSz,windowSz])
        for j in range(batchSz): # for each batch
            # fill the data and desired output vectors with text
            inputs[j,:] = new_input[j,i*windowSz:(i+1)*windowSz]
            answers[j,:] = new_input[j,i*windowSz+1:(i+1)*windowSz+1]
        # train on the examples
        _,outputs,next,Losses = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers})
        state = next # update the GRU state
        Loss+=Losses # add this batch's loss to total loss
        if (i+1) % 100 == 0 or i == number_of_windows - 1: # every 100 windows
            # retrieve learned weights
            embeddings,weights1,weights2,weights3,biases1,biases2,biases3 = sess.run([E,W1,W2,W3,b1,b2,b3])
            embeddings = np.asarray(embeddings)
            outputs = np.asarray(outputs)
            weights1 = np.asarray(weights1)
            weights2 = np.asarray(weights2)
            weights3 = np.asarray(weights3)
            biases1 = np.asarray(biases1)
            biases2 = np.asarray(biases2)
            biases3 = np.asarray(biases3)
            # define a list of the retrieved variables
            weights = ["embeddings","state","weights1","weights2","weights3","biases1","biases2","biases3"]
            # write them to file
            for variable in weights:
                np.savetxt(variable, eval(variable))
    # calculate training set perplexity
    train_perplexity[k] = np.exp(Loss/number_of_windows)
    print("Perplexity on the training set (Epoch " +str(k+1)+"): "+ str(train_perplexity[k]))
    ## test the network on development set
    # arrange the input in batches
    new_input = np.asarray(indexed_dev[0:batchSz*batchLen])
    new_input = np.reshape(new_input,(batchSz,batchLen))
    Devloss = 0 # reset the loss
    for i in range(number_of_windows): # for each window
        # initialize vectors for feeding data and desired output
        inputs = np.zeros([batchSz,windowSz])
        answers = np.zeros([batchSz,windowSz])
        for j in range(batchSz): # for each batch
            # fill the data and desired output vectors with text
            inputs[j,:] = new_input[j,i*windowSz:(i+1)*windowSz]
            answers[j,:] = new_input[j,i*windowSz+1:(i+1)*windowSz+1]
        # calculate loss
        DevLoss = sess.run(loss,feed_dict={inpt:inputs,answr:answers})
        Devloss+=DevLoss # add this set of batches' loss to total loss
    # calculate development set perplexity
    dev_perplexity[k] = np.exp(Devloss/number_of_windows)
    print("Perplexity on the development set (Epoch " +str(k+1)+"): "+ str(dev_perplexity[k]))
    ## if development set perplexity is increasing, stop training to prevent overfitting
    if k != 0 and dev_perplexity[k] > dev_perplexity[k-1]:
        break
# timer
print("Finished training at " + time.strftime('%l:%M%p'))
### test the network on test set ###
# number of windows and batch length for the development set
batchLen = int(floor(float((len(indexed_test)))/float(batchSz)))
number_of_windows = int(floor(float(batchLen)/float(windowSz)))
# arrange the input in batches
new_input = np.asarray(indexed_test[0:batchSz*batchLen])
new_input = np.reshape(new_input,(batchSz,batchLen))
Testloss = 0 # initialize loss
for i in range(number_of_windows): # for each window
    # initialize vectors for feeding data and desired output
    inputs = np.zeros([batchSz,windowSz])
    answers = np.zeros([batchSz,windowSz])
    for j in range(batchSz): # for each batch
        # fill the data and desired output vectors with text
        inputs[j,:] = new_input[j,i*windowSz:(i+1)*windowSz]
        answers[j,:] = new_input[j,i*windowSz+1:(i+1)*windowSz+1]
    # calculate loss
    TestLoss = sess.run(loss,feed_dict={inpt:inputs,answr:answers})
    Testloss+=TestLoss # add this set of batches' loss to total loss
# calculate test set perplexity
test_perplexity = np.exp(Testloss/number_of_windows)
print("Perplexity on the test set:" + str(test_perplexity))
# timer
print("Finishing time:" + time.strftime('%l:%M%p'))
