#!/usr/bin/python27
# -*- coding: utf-8 -*-

# NOTE: This file should be in the same directory as Combined_NN_Model.py or LDA_Model.py

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
nltk.download('wordnet')

### Global Set keys

set_key_list = ['train','dev','test']
indexes = {key: [] for key in set_key_list}
sets    = {key: [] for key in set_key_list}
lengths = {key: [] for key in set_key_list}
Max     = {key: [] for key in set_key_list}
vote     = {key: [] for key in set_key_list}
V = OrderedDict({}) # vocabulary

### define the preprocessing function to add padding and remove punctuation, special characters and stopwords (neural network)

def NN_clean(text,stop):
    # check input arguments for valid type
    assert type(text) is list or type(text) is str or type(text) is unicode
    assert type(stop) is set or type(stop) is list
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
        # determine the set of punctuations that should be removed
        exclude = set(string.punctuation)
        # remove punctuation
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

### define the preprocessing function to add padding and remove punctuation, special characters and stopwords (LDA)

# NOTE: Since LDA doesn't care about sentence structure, unlike NN_clean, the entire comment should be fed into this function as a continuous string
# NOTE: The Reddit dataset seems to encode the quote blocks as just new lines. Therefore, there is no way to get rid of quotes

def LDA_clean(text,stop):
    # check input arguments for valid type
    assert type(text) is unicode or type(text) is str
    assert type(stop) is set or type(stop) is list
    # remove apostrophes and replace with space
    text = text.replace("'"," ")
    # remove special characters
    special_free = ""
    for word in text.lower().split():
        # remove links
        if "http" not in word and "www" not in word:
            word = re.sub('[^A-Za-z0-9]+', ' ', word)
            special_free = special_free+" "+word
    # remove stopwords
    stop_free = " ".join([i for i in special_free.split() if i not in stop])
    # determine the set of punctuations that should be removed
    exclude = set(string.punctuation)
    # remove punctuation
    no_punc = re.compile('|'.join(map(re.escape, exclude)))
    punc_free = no_punc.sub(' ',stop_free)
    # lemmatize
    normalized = " ".join(nltk.stem.WordNetLemmatizer().lemmatize(word) for word in punc_free.split())
    return normalized

### define the relevance filters for gay marriage and marriage equality

def getFilterBasicRegex():
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)
GAYMAR = getFilterBasicRegex()

def getFilterEquRegex():
    return re.compile("^(?=.*marriage equality|.*equal marriage).*$", re.I)
MAREQU = getFilterEquRegex()

### define the parser

# NOTE: Parses for LDA if NN = False

def Parser(path,stop,vote_counting,NN):
    # if parsing for a neural network
    if NN == True:
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
            if NN == True: # if doing NN
                fout = open("nn_prep",'a+')
            else: # if doing LDA
                fout = open("lda_prep",'a+')
            # if we want to record sign of the votes
            if vote_counting == True:
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
                    if NN == True:
                        # tokenize the sentences
                        body = sent_detector.tokenize(body)
                        # clean the text for NN
                        body = NN_clean(body,stop)
                    else:
                        # clean the text for LDA
                        body = LDA_clean(body,stop)
                    # if the comment body is not empty after preprocessing
                    if NN == True:
                        if len(body) > 0:
                            counter += 1 # update the counter
                            # if we are interested in the sign of the votes
                            if vote_counting == True:
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
                    else: # if doing LDA
                        if body.strip() != "":
                            counter += 1 # update the counter
                            # if we are interested in the sign of the votes
                            if vote_counting == True:
                                # write the sign of the vote to file (-1 if negative, 0 if neutral, 1 if positive)
                                print(np.sign(int(comment["score"])),end="\n",file=vote)
                            # record the number of documents by year and month
                            created_at = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime('%Y-%m')
                            if created_at not in timedict:
                                timedict[created_at] = 1
                            else:
                                timedict[created_at] += 1
                            # remove mid-comment lines and set encoding
                            body = body.replace("\n","")
                            body = body.encode("utf-8")
                            # print the comment to file
                            print(" ".join(body.split()), sep=" ",end="\n", file=fout)
            # write the monthly cummulative number of comments to file
            print(counter,file=ccount)
            # close the files to save the data
            fin.close()
            fout.close()
            if NN == True:
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

### Function to call parser when needed and parse comments

def Parse_Rel_RC_Comments(path,stop,vote_counting, NN):
    # check input arguments for valid type
    if type(vote_counting) is not bool:
        raise Exception('Invalid vote counting argument')
    if type(NN) is not bool:
        raise Exception('Invalid NN argument')
    assert type(path) is str
    assert type(stop) is set or type(stop) is list
    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')
    # if preprocessed comments are available, ask if they should be rewritten
    if (NN == True and Path(path+"/nn_prep").is_file()) or (NN == False and Path(path+"/lda_prep").is_file()):
        Q = raw_input("Preprocessed comments are already available. Do you wish to delete them and parse anew [Y/N]?")
        if Q == 'Y' or Q == 'y':
            # delete previous preprocessed data
            if NN == True:
                os.remove(path+"/nn_prep")
                if Path(path+"/votes").is_file():
                    os.remove(path+"/votes")
            elif NN == False:
                os.remove(path+"/lda_prep")
            if Path(path+"/RC_Count_List").is_file():
                os.remove(path+"/RC_Count_List")
            if Path(path+"/RC_Count_Dict").is_file():
                os.remove(path+"/RC_Count_Dict")

            # check for the presence of data files
            if not glob.glob(path+'/*.bz2'):
                raise Exception('No data file found')

            # parse again
            # timer
            print("Started parsing at " + time.strftime('%l:%M%p'))
            Parser(path,stop,vote_counting,NN)

        else:
            print("Operation aborted")
            if not Path(path+"/RC_Count_List").is_file():
                raise Warning('Cummulative monthly counts not found. Please preprocess again')
            if NN == True:
                if not Path(path+"/votes").is_file():
                    raise Warning('Votes not found. Please preprocess again')
            if not Path(path+"/RC_Count_Dict").is_file():
                raise Warning('Monthly counts not found. Please preprocess again')
            pass
    else:
        if Path(path+"/RC_Count_List").is_file():
            os.remove(path+"/RC_Count_List")
        if NN == True:
            if Path(path+"/votes").is_file():
                os.remove(path+"/votes")
        if Path(path+"/RC_Count_Dict").is_file():
            os.remove(path+"/RC_Count_Dict")

        # check for the presence of data files
        if not glob.glob(path+'/*.bz2'):
            raise Exception('No data file found')

        # timer
        print("Started parsing at " + time.strftime('%l:%M%p'))
        Parser(path,stop,vote_counting,NN)

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

### Load, calculate or re-calculate the percentage of relevant comments/year

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

### function to determine comment indices for new training, development and test sets

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

    # sort the indices based on position in nn_prep
    for set_key in set_key_list:
        sets[set_key] = sorted(list(sets[set_key]))

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

### function for loading, calculating, or recalculating sets

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

### load or create vocabulary and load or create indexed versions of comments in sets

def Index_Set(path,set_key,MaxVocab,FrequencyFilter):

    # ensure the arguments have the correct type

    assert type(path) is str
    assert type(MaxVocab) is int
    assert type(FrequencyFilter) is int

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

        if set_key == 'train':

            # timer
            print("Started creating the dictionary at " + time.strftime('%l:%M%p'))

            ## initialize the vocabulary with various UNKs

            V.update({"*STOP2*":1,"*UNK*":2,"*UNKED*":3,"*UNKS*":4,"*UNKING*":5,"*UNKLY*":6,"*UNKER*":7,"*UNKION*":8,"*UNKAL*":9,"*UNKOUS*":10,"*STOP*":11})

        # read the dataset and index the relevant comments

        fin.seek(0)
        for counter,comm in enumerate(fin): # for each comment
            if counter in sets[set_key]: # if it belongs in the set
                comment = []
                for word in comm.split(): # for each word
                    if frequency[word] > FrequencyFilter: # filter non-frequent words
                        if word in V.keys(): # if the word is already in the vocabulary
                            comment.append(V[word]) # index it
                        elif set_key == 'train': # if the word is not in vocabulary and we are indexing the training set
                                if len(V)-11 <= MaxVocab: # if the vocabulary still has room (not counting STOPs and UNKs)
                                    V[word] = len(V)+1 # give it an index (leave index 0 for padding)
                                    comment.append(V[word]) # append it to the list of words

                                else: # if the vocabulary doesn't have room, assign the word to an UNK according to its suffix or lack thereof
                                    if word.endswith("ed"):
                                        comment.append(3)
                                    elif word.endswith("s"):
                                        comment.append(4)
                                    elif word.endswith("ing"):
                                        comment.append(5)
                                    elif word.endswith("ly"):
                                        comment.append(6)
                                    elif word.endswith("er"):
                                        comment.append(7)
                                    elif word.endswith("ion"):
                                        comment.append(8)
                                    elif word.endswith("al"):
                                        comment.append(9)
                                    elif word.endswith("ous"):
                                        comment.append(10)
                                    else: # if the word doesn't have any easily identifiable suffix
                                        comment.append(2)
                        else: # the word is not in vocabulary and we are not indexing the training set
                            if word.endswith("ed"):
                                comment.append(3)
                            elif word.endswith("s"):
                                comment.append(4)
                            elif word.endswith("ing"):
                                comment.append(5)
                            elif word.endswith("ly"):
                                comment.append(6)
                            elif word.endswith("er"):
                                comment.append(7)
                            elif word.endswith("ion"):
                                comment.append(8)
                            elif word.endswith("al"):
                                comment.append(9)
                            elif word.endswith("ous"):
                                comment.append(10)
                            else: # if the word doesn't have any easily identifiable suffix
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
    print("Finished indexing the "+set_key+" set at " + time.strftime('%l:%M%p'))

### turn votes into one-hot vectors

def One_Hot_Vote(vote_list):
    one_hot_vote = []
    for sign in vote_list:
        if int(sign) == -1:
            one_hot_vote.append([1,0,0])
        elif int(sign) == 0:
            one_hot_vote.append([0,1,0])
        elif int(sign) == 1:
            one_hot_vote.append([0,0,1])
        else:
            raise Exception('Votes could not be appended')
    return one_hot_vote

### import the correct labels for comment votes

def Get_Votes(path):
    if Path(path+"/votes").is_file():
        for set_key in set_key_list:
            vote[set_key] = []
        with open("votes",'r') as f:
            for index,sign in enumerate(f):
                sign = sign.strip()
                match_found = 0
                for set_key in set_key_list:
                    if index in sets[set_key]:
                        vote[set_key].append(sign)
                        match_found += 1
                if match_found == 0:
                    raise Exception('Votes could not be read from file')
        assert len(indexes['train']) == len(vote['train'])
        assert len(indexes['dev']) == len(vote['dev'])
        assert len(indexes['test']) == len(vote['test'])
    else:
        raise Exception('Labels for the sets could not be found')

    for set_key in set_key_list:
        vote[set_key] = One_Hot_Vote(vote[set_key])
    return vote['train'],vote['dev'],vote['test']
