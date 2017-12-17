#!/usr/bin/python27
# -*- coding: utf-8 -*-

from __future__ import print_function
from Utils import *

### determine hyperparameters

epochs = 3 # number of epochs
learning_rate = 0.003 # learning rate
MaxVocab = 100000 # maximum size of the vocabulary
batchSz = 50 # number of parallel batches
embedSz = 128 # embedding size
hiddenSz = 512 # number of units in the recurrent layer
ff1Sz = 1000 # number of units in the first feedforward layer
ff2Sz = 1000 # number of units in the second feedforward layer
keepP = 0.5 # 1 - dropout rate
alpha = 0.01 # L2 regularization constant
early_stopping = 1 # whether to stop training if development set perplexity is going up

### set default file encoding

reload(sys)
sys.setdefaultencoding('utf-8')

### Preprocessing

## determine the set of stopwords used in preprocessing

stop = set(nltk.corpus.stopwords.words('english'))

## determine the set of punctuations that should be removed

exclude = set(string.punctuation)

## where the data is

# NOTE: if not available, download from http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable accordingly

file_path = os.path.abspath(sys.argv[0])
path = os.path.dirname(file_path)

## call the parsing function

Parse_Rel_RC_Comments(path,stop,exclude,vote_counting=1,NN=1)

## call the function for calculating the percentage of relevant comments
# NOTE: RC_Count_Dict does not overwrite. If calling the function on a different subset of data than used before, delete the previous file manually

Perc_Rel_RC_Comment(path)

## Train neural network for language modeling

## where the output will be stored
# NOTE: record the variables most important to your iteration in the name

# output_path = path+"/"+"e"+str(epochs)+"_"+"hd"+"_"+str(hiddenSz)+"_"+"padded"
# if not os.path.exists(output_path):
#     print("Creating directory to store the output")
#     os.makedirs(output_path)
#
# ## create file to record the performance
#
# perf = open(output_path+"/Performance",'a+')
#
# ## create training, development and test sets
#
# # NOTE: Always index training set first.
# # NOTE: Maximum vocabulary size should not be changed in between the creation of various sets
#
# training_fraction = 0.80 # fraction of the data that is used for training
#
# # Determine the comments that will comprise each set
#
# Define_Sets(path,training_fraction)
#
# # Read and index the content of comments in each set
#
# print("Vocabulary size = " + str(MaxVocab),file=perf)
#
# for key in ['train','dev','test']:
#     if key == 'train':
#         V = Index_Set(path,key,MaxVocab)
#     else:
#         Index_Set(path,key,MaxVocab)
#
# # create the computation graph
#
# # Lang_Model_NN(V,learning_rate,batchSz,embedSz,hiddenSz,ff1Sz,ff2Sz,keepP,alpha,perf)
#
# # train and test the network
#
# # Train_Test_NN(epochs, batchSz,keepP,output_path, perf, early_stopping)
