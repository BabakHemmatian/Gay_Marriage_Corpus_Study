#!/usr/bin/python27
# -*- coding: utf-8 -*-
### import the required modules and functions


from __future__ import print_function
import tensorflow as tf
import numpy as np
import json
import os
import time
import datetime
import bz2
import string
import HTMLParser
import re
import sys
import nltk
import glob
from collections import defaultdict,OrderedDict
from pathlib2 import Path
from random import sample
from math import floor,ceil
nltk.download('stopwords')
nltk.download('punkt')

# Global Set keys
set_key_list = ['train','dev','test']
indexes = {key: [] for key in set_key_list}
sets    = {key: [] for key in set_key_list}
lengths = {key: [] for key in set_key_list}
Max     = {key: [] for key in set_key_list}
Loss    = {key: [] for key in set_key_list}
perplexity = {key: [] for key in set_key_list}

## define the preprocessing function to add padding and remove punctuation, special characters and stopwords
def clean(text,stop,exclude):
    # check input arguments for valid type
    assert type(text) is list or type(text) is str
    assert type(stop) is set or type(stop) is list
    assert type(exclude) is set or type(stop) is list
    # create a container for preprocessed sentences
    cleaned = []
    # iterate over the sentences
    for index,sent in enumerate(text):
        # remove apostrophes and replace
        sent = sent.replace("'"," ")
        # remove special characters
        special_free = ""
        for word in sent.lower().split():
            # remove links
            if "http" not in word and "www" not in word:
                word = re.sub('[^A-Za-z0-9]+', ' ', word)
                special_free = special_free+" "+word
        # remove stopwords
        stop_free = " ".join([i for i in special_free.split() if i not in stop])
        # remove punctuation --> instead of removing, we want to separate them, then add a stop character after those that signal end of a sentence. We should be fine disregarding the comment boundaries. If not, we could add a special character to be learned there too
        no_punc = re.compile('|'.join(map(re.escape, exclude)))
        punc_free = no_punc.sub(' ',stop_free)
        # add sentence and end of comment padding
        if punc_free.strip() != "":
            padded = punc_free+" *STOP*"
            if index+1 == len(text):
                padded = padded+" *STOP2*"
            cleaned.append(padded)
        elif punc_free.strip() == "" and len(text)!=1 and len(cleaned)!=0 and index+1 == len(text):
            cleaned[-1] = cleaned[-1]+" *STOP2*"
    return cleaned


## define the relevance filters
def getFilterBasicRegex():
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)
GAYMAR = getFilterBasicRegex()
def getFilterEquRegex():
    return re.compile("^(?=.*marriage equality|.*equal marriage).*$", re.I)
MAREQU = getFilterEquRegex()

def NN_Parser(path,stop,exclude,vote_counting):
    ## import the pre-trained PUNKT tokenizer for determining sentence boundaries
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # json parser
    decoder = json.JSONDecoder(encoding='utf-8')
    ## iterate over files in directory to preprocess the text and record the votes
    # initialize container for number of comments and indices related to each month
    timedict = dict()
    # counting the number of all processed comments
    counter = 0
    for filename in sorted(os.listdir(path)):
        # only include relevant files
        if os.path.splitext(filename)[1] == '.bz2' and 'RC_' in filename:
            ## prepare files
            # open the file as a text file, in utf8 encoding
            fin = bz2.BZ2File(filename,'r')
            # create a file to write the processed text to
            fout = open("nn_prep",'a+')
            # if we want to record sign of the votes
            if vote_counting == 1:
                # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
                vote = open("votes",'a+')
            # create a file to store the relevant indices for each month
            ccount = open("RC_Count_List",'a+')
            ## read data
            # for each comment
            for line in fin:
                # parse the json, and turn it into regular text
                comment = decoder.decode(line)
                body = HTMLParser.HTMLParser().unescape(comment["body"])
                # filter comments by relevance to the topic
                if len(GAYMAR.findall(body)) > 0 or len(MAREQU.findall(body)) > 0:
                    ## preprocess the comments
                    # tokenize the sentences
                    body = sent_detector.tokenize(body)
                    # clean the the text,
                    body = clean(body,stop,exclude)
                    # if the comment body is not empty after preprocessing
                    if body != []:
                        counter += 1 # update the counter
                        # if we are interested in the sign of the votes
                        if vote_counting == 1:
                            # write the sign of the vote to file (-1 if negative, 0 if neutral, 1 if positive)
                            print(np.sign(int(comment["score"])),end="\n",file=vote)
                            # record the number of documents by year and month
                        created_at = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime('%Y-%m')
                        if created_at not in timedict:
                            timedict[created_at] = 1
                        else:
                            timedict[created_at] += 1
                        for sen in body: # for each sentence in the comment
                            # remove mid-comment lines and set encoding
                            sen = sen.replace("\n","")
                            sen = sen.encode("utf-8")
                            # print the processed sentence to file
                            print(" ".join(sen.split()), end=" ", file=fout)
                        # ensure that each comment is on a separate line
                        print("\n",end="",file=fout)
            # write the monthly cummulative number of comments to file
            print(counter,file=ccount)
            # close the files to save the data
            fin.close()
            fout.close()
            vote.close()
            ccount.close()
            # timer
            print("Finished parsing "+filename+" at " + time.strftime('%l:%M%p'))
    # timer
    print("Finished parsing at " + time.strftime('%l:%M%p'))
    ## distribution of comments by month
    # if not already available, write the distribution to file
    if Path(path+"/RC_Count_Dict").is_file():
        os.remove(path+"/RC_Count_Dict")
    fcount = open("RC_Count_Dict",'a+')
    for month,docs in timedict.iteritems():
        print(month+" "+str(docs),end='\n',file=fcount)
    fcount.close

### parse data
def Parse_Rel_RC_Comments(path,stop,exclude,vote_counting, NN):
    # check input arguments for valid type
    if vote_counting != 0 and vote_counting != 1:
        raise Exception('Invalid vote counting argument')
    if NN != 0 and NN != 1:
        raise Exception('Invalid NN argument')
    assert type(path) is str
    assert type(stop) is set or type(stop) is list
    assert type(exclude) is set or type(stop) is list
    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')

    if Path(path+"/nn_prep").is_file():
        Q = raw_input("Preprocessed comments are already available. Do you wish to delete them and parse anew [Y/N]?")
        if Q == 'Y' or Q == 'y':
            # delete previous preprocessed data
            os.remove(path+"/nn_prep")
            if Path(path+"/RC_Count_List").is_file():
                os.remove(path+"/RC_Count_List")
            if Path(path+"/votes").is_file():
                os.remove(path+"/votes")
            if Path(path+"/RC_Count_Dict").is_file():
                os.remove(path+"/RC_Count_Dict")

            if not glob.glob(path+'/*.bz2'):
                raise Exception('No data file found')

            if NN == 1:
                # timer
                print("Started parsing at " + time.strftime('%l:%M%p'))
                NN_Parser(path,stop,exclude,vote_counting)

        else:
            print("Operation aborted")
            if not Path(path+"/RC_Count_List").is_file():
                raise Warning('Cummulative monthly counts not found. Please preprocess again')
            if not Path(path+"/votes").is_file():
                raise Warning('Votes not found. Please preprocess again')
            if not Path(path+"/RC_Count_Dict").is_file():
                raise Warning('Monthly counts not found. Please preprocess again')
            pass
    else:
        if Path(path+"/RC_Count_List").is_file():
            os.remove(path+"/RC_Count_List")
        if Path(path+"/votes").is_file():
            os.remove(path+"/votes")
        if Path(path+"/RC_Count_Dict").is_file():
            os.remove(path+"/RC_Count_Dict")

        if NN == 1:
            # timer
            print("Started parsing at " + time.strftime('%l:%M%p'))
            NN_Parser(path,stop,exclude,vote_counting)

### determine what percentage of the posts in each year was relevant
def Rel_Counter(path):

    if not Path(path+"/RC_Count_Dict").is_file():
        raise Exception('Monthly counts cannot be found')
    if not Path(path+"/RC_Count_Total").is_file():
        raise Exception('Total monthly comment counts cannot be found')

    with open("RC_Count_Dict",'r') as f:
        timedict = dict()
        for line in f:
            if line.strip() != "":
                (key, val) = line.split()
                timedict[key] = int(val)
    # load the total monthly counts into a dictionary
    d = {}
    with open("RC_Count_Total",'r') as f:
        for line in f:
            line = line.replace("\n","")
            if line.strip() != "":
                (key, val) = line.split("  ")
                d[key] = int(val)
    # calculate the total yearly counts
    total_year = {}
    for keys in d:
        if str(keys[3:7]) in total_year:
            total_year[str(keys[3:7])] += d[keys]
        else:
            total_year[str(keys[3:7])] = d[keys]
    # calculate the yearly relevant counts
    relevant_year = {}
    for key in timedict:
        if str(key[:4]) in relevant_year:
            relevant_year[str(key[:4])] += timedict[key]
        else:
            relevant_year[str(key[:4])] = timedict[key]
    # calculate the percentage of comments in each year that was relevant and write it to file
    perc_rel = {}
    rel = open("perc_rel",'a+')
    for key in relevant_year:
        perc_rel[key] = float(relevant_year[key]) / float(total_year[key])
    print(sorted(perc_rel.items()),file=rel)
    rel.close

def Perc_Rel_RC_Comment(path):
    if Path(path+"/perc_rel").is_file():
        Q = raw_input("Calculated percentages are already available. Do you wish to delete them and count anew [Y/N]?")
        if Q == 'Y' or Q == 'y':
            os.remove(path+"/perc_rel")
            Rel_Counter(path)
        else:
            print("Operation aborted")
            pass
    else:
        Rel_Counter(path)


def Create_New_Sets(path,training_fraction,timelist):

    print("Creating sets")

    # determine indices of set elements

    num_comm = timelist[-1] # number of comments

    num_train = int(ceil(training_fraction * num_comm)) # size of training set
    sets['train'] = sample(range(num_comm),num_train) # choose training comments at random

    remaining = [x for x in range(num_comm) if x not in sets['train']]
    num_dev = int(floor(len(remaining)/2)) # size of development set
    sets['dev']  = sample(remaining,num_dev) # choose development comments at random
    sets['test'] = [x for x in remaining if x not in sets['dev']] # use the rest as test set

    # Check dev and test sets came out with right proportions
    assert (len(sets['dev']) - len(sets['test'])) <= 1
    assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == timelist[-1]

    # write the sets to file
    for set_key in set_key_list:
        with open(set_key+'_set','a+') as f:
            for index in sets[set_key]:
                print(index,end='\n',file=f)

    # ensure set sizes are correct

    assert len(sets['dev']) - len(sets['test']) < 1
    assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == timelist[-1]



def Define_Sets(path,training_fraction):

    # ensure the arguments have the correct types and values

    assert type(path) is str
    assert 0 < training_fraction and 1 > training_fraction

    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')

    # load the number of comments
    timelist = []
    if Path(path+"/RC_Count_List").is_file():
        with open("RC_Count_List",'r') as f:
            for line in f:
                timelist.append(int(line))
    else:
        raise Exception("The monthly counts could not be found")

    # if indexed comments are available
    if Path(path+"/indexed_train").is_file() and Path(path+"/indexed_dev").is_file() and Path(path+"/indexed_test").is_file():
        # determine if the comments and their relevant indices should be deleted and re-initialized or the sets should just be loaded
        Q = raw_input("Indexed comments are already available. Do you wish to delete sets and create new ones [Y/N]?")

        # If recreating the sets is requested, delete the current ones and reinitialize

        if Q == "Y" or Q == "y":
            print("Deleting any existing sets and indexed comments")
            for set_key in set_key_list:
                if Path(path+"/indexed_"+set_key).is_file():
                    os.remove(path+"/indexed_"+set_key)
                if Path(path+"/"+set_key+"_set").is_file():
                    os.remove(path+"/"+set_key+"_set")

            Create_New_Sets(path,training_fraction,timelist)

        # If recreating is not requested, attempt to load the sets

        elif Q == "N" or Q == "n":

            # if the sets are found, load them

            if Path(path+"/train_set").is_file() and Path(path+"/dev_set").is_file() and Path(path+"/test_set").is_file():

                print("Loading sets from file")

                for set_key in set_key_list:
                    with open(set_key + '_set','r') as f:
                        for line in f:
                            if line.strip() != "":
                                sets[set_key].append(int(line))
                    sets[set_key] = np.asarray(sets[set_key])

                    # ensure set sizes are correct

                    assert len(sets['dev']) - len(sets['test']) < 1
                    assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == timelist[-1]

            # if the sets cannot be found, delete any current sets and create new sets

            else:

                print("Failed to load previous sets. Reinitializing")

                for set_key in set_key_list:
                    if Path(path+"/indexed_"+set_key).is_file():
                        os.remove(path+"/indexed_"+set_key)
                    if Path(path+"/"+set_key+"_set").is_file():
                        os.remove(path+"/"+set_key+"_set")

                Create_New_Sets(path,training_fraction,timelist)

        else: # if response was something other tha Y or N, try again
            print("Operation aborted")
            pass

    else: # no indexed comments available
    # if not (Path(path+"/indexed_train").is_file() and Path(path+"/indexed_dev").is_file() and Path(path+"/indexed_test").is_file()): # no indexed comments available

        # delete any possible partial indexed set

        for set_key in set_key_list:
            if Path(path+"/indexed_"+set_key).is_file():
                os.remove(path+"/indexed_"+set_key)

        # check to see if there are sets available, if so load them

        if Path(path+"/train_set").is_file() and Path(path+"/dev_set").is_file() and Path(path+"/test_set").is_file():

            print("Loading sets from file")

            for set_key in set_key_list:
                with open(set_key + '_set','r') as f:
                    for line in f:
                        if line.strip() != "":
                            sets[set_key].append(int(line))
                sets[set_key] = np.asarray(sets[set_key])

                # ensure set sizes are correct

            assert len(sets['dev']) - len(sets['test']) < 1
            assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == timelist[-1]

        else: # if not all sets are found

        # delete any partial set

            for set_key in set_key_list:
                if Path(path+"/indexed_"+set_key).is_file():
                    os.remove(path+"/indexed_"+set_key)
                if Path(path+"/"+set_key+"_set").is_file():
                    os.remove(path+"/"+set_key+"_set")

            # create new sets

            Create_New_Sets(path,training_fraction,timelist)

def Index_Set(path,set_key,MaxVocab):

    # ensure the arguments have the correct type

    assert type(path) is str
    assert type(MaxVocab) is int

    # check the given path

    if not os.path.exists(path):
        raise Exception('Invalid path')

    ## record word frequency in the entire dataset
    frequency = defaultdict(int)

    if Path(path+"/nn_prep").is_file():

        fin = open('nn_prep','r')
        for comment in fin: # for each comment
            for token in comment.split(): # for each word
                frequency[token] += 1 # count the number of occurrences
    else:
        raise Exception('Pre-processed dataset could not be found')

    # if indexed comments are available and we are trying to index the training set

    if Path(path+"/indexed_"+set_key).is_file() and set_key == 'train':

        # If the vocabulary is available, load it

        if Path(path+"/dict").is_file():
            print("Loading dictionary from file")
            V = OrderedDict({})

            with open("dict",'r') as f:
                for line in f:
                    if line.strip() != "":
                        (key, val) = line.split()
                        V[key] = int(val)

        else: # if the vocabulary is not available

            # delete the (possible) dictionary-less indexed training set file

            if Path(path+"/indexed_"+set_key).is_file():
                os.remove(path+"/indexed_"+set_key)

    # if indexed comments are available, load them

    if Path(path+"/indexed_"+set_key).is_file():
        print("Loading the set from file")

        with open("indexed_"+set_key,'r') as f:
            for line in f:
                assert line.strip() != ""
                comment = []
                for index in line.split():
                    comment.append(index)
                indexes[set_key].append(comment)

    else: # if the indexed comments are not available, create them

        unk_suffixes = ['ed', 's', 'ing', 'ly', 'er', 'ion', 'al', 'ous']

        if set_key == 'train':

            # timer
            print("Started creating the dictionary at " + time.strftime('%l:%M%p'))

            ## initialize the vocabulary with various UNKs
            V = OrderedDict({"*STOP*":0,"*STOP2*":1,"*UNK*":2,"*UNKED*":3,"*UNKS*":4,"*UNKING*":5,"*UNKLY*":6,"*UNKER*":7,"*UNKION*":8,"*UNKAL*":9,"*UNKOUS*":10})

        # read the dataset and index the relevant comments

        fin.seek(0)
        for counter,comm in enumerate(fin): # for each comment
            if counter in sets[set_key]: # if it belongs in the set
                comment = []
                for word in comm.split(): # for each word
                    if frequency[word] > 5: # filter non-frequent words
                        if word in V.keys(): # if the word is already in the vocabulary
                            comment.append(V[word]) # index it
                        elif set_key == 'train': # if the word is not in vocabulary and we are indexing the training set
                                if len(V)-11 <= MaxVocab: # if the vocabulary still has room (not counting STOPs and UNKs)
                                    V[word] = len(V) # give it an index
                                    comment.append(V[word]) # append it to the list of words

                                else: # if the vocabulary doesn't have room, assign the word to an UNK according to its suffix or lack thereof
                                    for suffix in unk_suffixes: # check for any recognizable suffix
                                        if word.endswith(suffix):
                                            suffix_counter = 0
                                            for key in V.iterkeys():
                                                suffix_counter += 1
                                                if suffix_counter <= 11:
                                                    if suffix in key:
                                                        comment.append(V[key])
                                                        break
                                                if suffix_counter > 11:
                                                    break
                                        else: # does not have any recognizable suffix
                                            comment.append(2)
                        else: # the word is not in vocabulary and we are not indexing the training set
                            for suffix in unk_suffixes: # check for any recognizable suffix
                                if word.endswith(suffix):
                                    suffix_counter = 0
                                    for key in V.iterkeys():
                                        suffix_counter += 1
                                        if suffix_counter <= 11:
                                            if suffix in key:
                                                comment.append(V[key])
                                                break
                                        if suffix_counter > 11:
                                            break
                                else: # does not have any recognizable suffix
                                    comment.append(2)
                indexes[set_key].append(comment)
        # save the vocabulary to file
        if set_key == 'train':
            vocab = open("dict",'a+')
            for word,index in V.iteritems():
                print(word+" "+str(index),file=vocab)
            vocab.close
        # save the indices for training set to file
        with open("indexed_"+set_key,'a+') as f:
            for comment in indexes[set_key]:
                assert len(comment) != 0
                for ind,word in enumerate(comment):
                    if ind != len(comment) - 1:
                        print(word,end=" ",file=f)
                    elif ind == len(comment) - 1:
                        print(word,file=f)

        assert len(indexes[set_key]) == len(sets[set_key])
    # timer
    print("Finished indexing the"+set_key+" set at " + time.strftime('%l:%M%p'))
    if set_key == 'train':
        return V

def Lang_Model_NN(V,learning_rate,batchSz,embedSz,hiddenSz,ff1Sz,ff2Sz,keepP,alpha,perf):
    ## record the hyperparameters
    print("Learning_rate = " + str(learning_rate),file=perf)
    print("Batch size = " + str(batchSz),file=perf)
    print("Embedding size = " + str(embedSz),file=perf)
    print("Recurrent layer size = " + str(hiddenSz),file=perf)
    print("1st feedforward layer size = " + str(ff1Sz),file=perf)
    print("2nd feedforward layer size = " + str(ff2Sz),file=perf)
    print("Dropout rate = " + str(1 - keepP),file=perf)
    print("L2 regularization constant = " + str(alpha),file=perf)
    ### set up the computation graph ###
    ### create placeholders for input, output
    inpt = tf.placeholder(tf.int32, shape=[None,None])
    answr = tf.placeholder(tf.int32, shape=[None,None])
    loss_weight = tf.placeholder(tf.float32, shape=[None,None])
    DOutRate = tf.placeholder(tf.float32)
    ### set up the variables
    # initial embeddings
    E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))
    # look up the embeddings
    embed = tf.nn.embedding_lookup(E, inpt)
    sum_weights = tf.nn.l2_loss(embed)
    # define the recurrent layer (Gated Recurrent Unit)
    rnn= tf.contrib.rnn.GRUCell(hiddenSz)
    initialState = rnn.zero_state(batchSz, tf.float32)
    output, nextState = tf.nn.dynamic_rnn(rnn, embed,initial_state=initialState)
    sum_weights = sum_weights + tf.nn.l2_loss(nextState)
    # create weights and biases for three feedforward layers
    W1 = tf.Variable(tf.random_normal([hiddenSz,ff1Sz], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(W1)
    b1 = tf.Variable(tf.random_normal([ff1Sz], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(b1)
    l1logits = tf.nn.relu(tf.tensordot(output,W1,[[2],[0]])+b1)
    l1Output = tf.nn.dropout(l1logits,DOutRate) # apply dropout
    W2 = tf.Variable(tf.random_normal([ff1Sz,ff2Sz], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(W2)
    b2 = tf.Variable(tf.random_normal([ff2Sz], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(b2)
    l2Output = tf.nn.relu(tf.tensordot(l1Output,W2,[[2],[0]])+b2)
    W3 = tf.Variable(tf.random_normal([ff2Sz,len(V)], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(W3)
    b3 = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
    sum_weights = sum_weights + tf.nn.l2_loss(b3) # add to loss to get L2 regularization
    ### calculate loss
    # calculate logits
    logits = tf.tensordot(l2Output,W3,[[2],[0]])+b3
    # calculate sequence cross-entropy loss
    xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=answr,weights=loss_weight)
    loss = tf.reduce_mean(xEnt) # + (alpha * sum_weights)
    ### training with AdamOptimizer
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def Test_NN(batchSz,Max_l,keepP,output_path,perf,set_key):
    Loss[set_key] = 0 # reset the loss
    # initialize vectors for feeding data and desired output
    inputs = np.zeros([batchSz,Max_l])
    answers = np.zeros([batchSz,Max_l])
    loss_weights = np.zeros([batchSz,Max_l])
    j = 0 # batch comment counter
    p = 0 # batch counter
    for i in range(len(indexes[set_key])):
        inputs[j,:lengths[set_key][i]] = indexes[set_key][i]
        answers[j,:lengths[set_key][i]-1] = indexes[set_key][i][1:]
        loss_weights[j,:lengths[set_key][i]] = 1
        j += 1
        if j == batchSz - 1:
            # train on the examples if the set is the training set
            if set_key == 'train':
                _,outputs,next,Loss[set_key] = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:keepP})
            else: # if the set is not the training set
                Loss[set_key] = sess.run(loss,feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:1})
            j = 0
            p += 1
            inputs = np.zeros([batchSz,Max_l])
            answers = np.zeros([batchSz,Max_l])
            weighting = np.zeros([batchSz,Max_l])
            state = next # update the GRU state
            Loss+=Losses # add this batch's loss to total loss
        if set_key == 'train':
            if (i+1) % 10000 == 0 or i == len(indexed_train) - 1: # every 10000 comments or at the end of training, save the weights
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
                    np.savetxt(output_path+"/"+variable, eval(variable))
    # calculate training set perplexity
    train_perplexity[k] = np.exp(Loss/p)
    print("Perplexity on the " + set_key + "set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]))
    print("Perplexity on the " + set_key + "set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]),file=perf)
    return

### training the network ###
def Train_Test_NN(epochs, batchSz,keepP,output_path, perf, early_stopping):

    assert early_stopping == 1 or early_stopping == 0

    ### create the session and initialize variables

    config = tf.ConfigProto(device_count = {'GPU': 0}) # Use only CPU (large matrices)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    state = sess.run(initialState)
    ## initialize vectors to store set perplexities
    for set_key in set_key_list:
        perplexity[set_key] = np.empty(epochs)
    ### train the network
    ## create a list of training comment lengths
    for set_key in set_key_list:
        for i,x in enumerate(indexes[set_key]):
            lengths.append(len(indexes[set_key][i]))
        Max[set_key] = max(lengths[set_key]) # maximum length of a comment in this set
    Max_l = max(Max['train'],Max['dev'],Max['test']) # max length of a comment in the whole dataset
    # for each epoch
    print("Number of epochs: "+str(epochs))
    print("Number of epochs: "+str(epochs),file=perf)
    k = 0 # epoch counter
    # timer
    print("Started epoch "+str(k+1)+" at "+time.strftime('%l:%M%p'))
    ## train on the training set and calculate perplexity on all sets
    for set_key in set_key_list:

        Test_NN(batchSz,Max_l,keepP,output_path,perf,set_key)

        if set_key == 'dev':
            if early_stopping == 1:
                if perplexity[set_key][k] > perplexity[set_key][k-1]:
                    k == epochs - 1
    if k == epochs - 1:
        pass



    #     ## test the network on development set
    #     Devloss = 0 # reset the loss
    #     # initialize vectors for feeding data and desired output
    #     inputs = np.zeros([batchSz,Max_l])
    #     answers = np.zeros([batchSz,Max_l])
    #     weighting = np.zeros([batchSz,Max_l])
    #     j = 0 # batch comment counter
    #     p = 0 # batch counter
    #     for i in range(len(indexed_dev)): # for each comment
    #         inputs[j,:dev_lengths[i]] = indexed_dev[i]
    #         if len(indexed_dev[i]) == 1:
    #             continue
    #         else:
    #             answers[j,:dev_lengths[i]-1] = indexed_dev[i][1:]
    #         weighting[j,:dev_lengths[i]] = 1
    #         j += 1
    #         if j == batchSz - 1:
    #             # calculate loss
    #             DevLoss = sess.run(loss,feed_dict={inpt:inputs,answr:answers,loss_weight:weighting,DOutRate:1})
    #             Devloss+=DevLoss # add this set of batches' loss to total loss
    #             j = 0
    #             p += 1
    #             inputs = np.zeros([batchSz,Max_l])
    #             answers = np.zeros([batchSz,Max_l])
    #             weighting = np.zeros([batchSz,Max_l])
    #     # calculate development set perplexity
    #     dev_perplexity[k] = np.exp(Devloss/p)
    #     print("Perplexity on the development set (Epoch " +str(k+1)+"): "+ str(dev_perplexity[k]))
    #     print("Perplexity on the development set (Epoch " +str(k+1)+"): "+ str(dev_perplexity[k]),file=perf)
    #     ## if development set perplexity is increasing, stop training to prevent overfitting
    #     if k != 0 and dev_perplexity[k] > dev_perplexity[k-1]:
    #         break
    # # timer
    # print("Finished training at " + time.strftime('%l:%M%p'))
    # ### test the network on test set ###
    # Testloss = 0 # initialize loss
    # # initialize vectors for feeding data and desired output
    # inputs = np.zeros([batchSz,Max_l])
    # answers = np.zeros([batchSz,Max_l])
    # weighting = np.zeros([batchSz,Max_l])
    # j = 0 # batch comment counter
    # p = 0 # batch counter
    # for i in range(len(indexed_test)):
    #     inputs[j,:test_lengths[i]] = indexed_test[i]
    #     if len(indexed_test[i]) == 1:
    #         continue
    #     else:
    #         answers[j,:test_lengths[i]-1] = indexed_test[i][1:]
    #     weighting[j,:test_lengths[i]] = 1
    #     j += 1
    #     if j == batchSz - 1:
    #         # calculate loss
    #         testLoss = sess.run(loss,feed_dict={inpt:inputs,answr:answers,loss_weight:weighting,DOutRate:1})
    #         Testloss+=testLoss # add this set of batches' loss to total loss
    #         j = 0
    #         p += 1
    #         inputs = np.zeros([batchSz,Max_l])
    #         answers = np.zeros([batchSz,Max_l])
    #         weighting = np.zeros([batchSz,Max_l])
    # # calculate test set perplexity
    # test_perplexity = np.exp(Testloss/p)
    # print("Perplexity on the test set:" + str(test_perplexity))
    # print("Perplexity on the test set:" + str(test_perplexity),file=perf)
    # # timer
    # print("Finishing time:" + time.strftime('%l:%M%p'))
    # # close the performance file
    # perf.close()
