#!/usr/bin/python27
# -*- coding: utf-8 -*-

# NOTE: This file should be in the same directory as Combined_NN_Model.py or New_LDA_Analysis.py

### import the required modules and functions

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gensim
import multiprocessing
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
import csv
from collections import defaultdict,OrderedDict
from pathlib2 import Path
from random import sample
from math import floor,ceil
from scipy.sparse import csr_matrix
from functools import partial
from contextlib import contextmanager
from threading import Thread, Lock
import hashlib
import tzlocal
import subprocess
import pickle
nltk.download('punkt')
nltk.download('wordnet')
from config import *

### Global Set keys

set_key_list = ['train','dev','test'] # for NN
LDA_set_keys = ['train','eval'] # for LDA

sets    = {key: [] for key in set_key_list} # for NN
indexes = {key: [] for key in set_key_list}
LDA_sets = {key: [] for key in LDA_set_keys} # for LDA
lengths = {key: [] for key in set_key_list}
Max     = {key: [] for key in set_key_list}
vote     = {key: [] for key in set_key_list} # for NN
V = OrderedDict({}) # vocabulary

### Function for writing parameters and model performance to file

def Write_Performance(output_path=output_path):
    with open(output_path+"/Performance",'a+') as perf:
        if NN==False:
            print("***",file=perf)
            print("Time: "+time.strftime("%Y-%m-%d %H:%M:%S"),file=perf)
            print("*** Hyperparameters ***", file=perf)
            print("Training fraction = " + str(training_fraction),file=perf)
            print("Maximum vocabulary size = " + str(MaxVocab),file=perf)
            print("Minimum number of documents a token can appear in and be included = " + str(no_below),file=perf)
            print("Fraction of documents, tokens appearing more often than which will be filtered out = " + str(no_above),file=perf)
            print("Number of topics = " + str(num_topics),file=perf)
            print("Fraction of topics sampled = " + str(sample_topics),file=perf)
            print("Number of top words recorded for each topic = " + str(topn),file=perf)
            print("Number of comments sampled from each top topic = " + str(sample_comments),file=perf)
            print("Minimum comment length for sampled comments = " + str(min_comm_length),file=perf)
            print("Alpha (LDA) = " + str(alpha),file=perf)
            print("Eta (LDA) = " + str(eta),file=perf)
            print("Minimum topic probability = " + str(minimum_probability),file=perf)
            print("Minimum term probability = " + str(minimum_phi_value),file=perf)

        ## TODO: Write a separate set of variables to file for NN

### Functions for data file retrieval

## Raw Reddit data filename format

def _get_rc_filename(yr,mo):
    if len(str(mo))<2:
        mo='0{}'.format(mo)
    assert len(str(yr))==4
    assert len(str(mo))==2
    return 'RC_{}-{}.bz2'.format(yr, mo)

## Download Reddit comment data

def download(year, month, path):
    BASE_URL='https://files.pushshift.io/reddit/comments/'
    url=BASE_URL+_get_rc_filename(year, month)
    print ('Sending request to {}.'.format(url))
    os.system('cd {} && wget {}'.format(path, url))

## Get Reddit compressed data file hashsums to check downloaded files' integrity

def Get_Hashsums(path):
    # notify the user
    print ('Retrieving hashsums to check file integrity')
    # set the URL to download hashsums from
    url='https://files.pushshift.io/reddit/comments/sha256sum.txt'
    # remove any old hashsum file
    if Path(path+'/sha256sum.txt').is_file():
        os.remove(path+'/sha256sum.txt')
    # download hashsums
    os.system('cd {} && wget {}'.format(path, url))
    # retrieve the correct hashsums
    hashsums = {}
    with open(path+'/sha256sum.txt','rb') as f:
        for line in f:
            if line.strip() != "":
                (val, key) = str(line).split()
                hashsums[key] = val

    return hashsums

## calculate hashsums for downloaded files in chunks of size 4096B

def sha256(fname, path=path):
    hash_sha256= hashlib.sha256()
    with open("{}/{}".format(path, fname), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

### define the preprocessing function to add padding and remove punctuation, special characters and stopwords (neural network)

def NN_clean(text,stop):

    # check input arguments for valid type
    assert type(text) is list or type(text) is str or type(text) is unicode
    assert type(stop) is set or type(stop) is list

    # create a container for preprocessed sentences
    cleaned = []

    for index,sent in enumerate(text): # iterate over the sentences

        # remove stopwords --> check to see if apostrophes are properly encoded
        stop_free = " ".join([i for i in sent.lower().split() if i.lower() not in stop])
        replace = {"should've":"should","mustn't":"mustn","shouldn't":"shouldn","couldn't":"couldn","shan't":"shan", "needn't":"needn", "-":""}
        substrs = sorted(replace, key=len, reverse=True)
        regexp = re.compile('|'.join(map(re.escape, substrs)))
        stop_free = regexp.sub(lambda match: replace[match.group(0)], stop_free)

        # remove special characters
        special_free = ""
        for word in stop_free.split():
            if "http" not in word and "www" not in word: # remove links
                word = re.sub('[^A-Za-z0-9]+', ' ', word)
                if word.strip() != "":
                    special_free = special_free+" "+word.strip()

        # check for stopwords again
        special_free = " ".join([i for i in special_free.split() if i not in stop])

        # add sentence and end of comment padding
        if special_free.strip() != "":
            padded = punc_free+" *STOP*"
            if index+1 == len(text):
                padded = padded+" *STOP2*"
            cleaned.append(padded)
        elif special_free.strip() == "" and len(text)!=1 and len(cleaned)!=0 and index+1 == len(text):
            cleaned[-1] = cleaned[-1]+" *STOP2*"

    return cleaned

### define the preprocessing function to lemmatize, and remove punctuation, special characters and stopwords (LDA)

# NOTE: Since LDA doesn't care about sentence structure, unlike NN_clean, the entire comment should be fed into this function as a continuous string
# NOTE: The Reddit dataset seems to encode the quote blocks as just new lines. Therefore, there is no way to get rid of quotes

def LDA_clean(text,stop):

    # check input arguments for valid type
    assert type(text) is unicode or type(text) is str
    assert type(stop) is set or type(stop) is list

    # remove stopwords
    stop_free = " ".join([i for i in text.lower().split() if i.lower() not in stop])
    replace = {"should've":"should","mustn't":"mustn","shouldn't":"shouldn","couldn't":"couldn","shan't":"shan", "needn't":"needn", "-":""}
    substrs = sorted(replace, key=len, reverse=True)
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    stop_free = regexp.sub(lambda match: replace[match.group(0)], stop_free)

    # remove special characters
    special_free = ""
    for word in stop_free.lower().split():
        if "http" not in word and "www" not in word: # remove links
            word = re.sub('[^A-Za-z0-9]+', ' ', word)
            if word.strip() != "":
                special_free = special_free+" "+word.strip()

    # check for stopwords again
    special_free = " ".join([i for i in special_free.split() if i not in stop])

    # lemmatize
    normalized = " ".join([nltk.stem.WordNetLemmatizer().lemmatize(word) if word != "us" else "us" for word in special_free.split()])

    return normalized

### define the relevance filters for gay marriage and marriage equality

def getFilterBasicRegex():
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)

GAYMAR = getFilterBasicRegex()

def getFilterEquRegex():
    return re.compile("^(?=.*marriage equality|.*equal marriage).*$", re.I)

MAREQU = getFilterEquRegex()

def get_parser_fns(year=None, month=None, path=path):
    assert ( ( isinstance(year, type(None)) and isinstance(month, type(None)) ) or
             ( not isinstance(year, type(None)) and not isinstance(month, type(None)) ) )
    if isinstance(year, type(None)) and isinstance(month, type(None)):
        suffix=""
    else:
        suffix="-{}-{}".format(year, month)
    fns=dict((("lda_prep","{}/lda_prep{}".format(path, suffix)),
              ("original_comm","{}/original_comm{}".format(path, suffix)),
              ("original_indices","{}/original_indices{}".format(path, suffix)),
              ("votes","{}/votes{}".format(path, suffix)),
              ("counts","{}/RC_Count_List{}".format(path, suffix)),
              ("timedict","{}/RC_Count_Dict{}".format(path, suffix))
             ))
    if NN:
        fns["nn_prep"]="{}/nn_prep{}".format(path, suffix)
    return fns
 
def parse_one_month_wrapper(args):
    parse_one_month(*args)

def parse_one_month(year, month, hashsums, path=path, stop=stop,
                    vote_counting=vote_counting, NN=NN,
                    write_original=WRITE_ORIGINAL, download_raw=DOWNLOAD_RAW,
                    clean_raw=CLEAN_RAW):
    timedict=dict()

    if NN == True: # if parsing for a neural network
        ## import the pre-trained PUNKT tokenizer for determining sentence boundaries
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    decoder = json.JSONDecoder(encoding='utf-8') 
 
    ## prepare files
    filename=_get_rc_filename(year, month) # get the relevant compressed data file name
    if not filename in os.listdir(path) and download_raw: # if the file is not available on disk and download is turned on
        download(year, month, path) # download the relevant file
        # check data file integrity and download again if needed
        filesum = sha256(filename) # calculate hashsum for the data file on disk
        attempt = 0 # number of hashsum check trials for the current file
        while filesum != hashsums[filename]: # if the file hashsum does not match the correct hashsum
            attempt += 1 # update hashsum check counter
            if attempt == 5: # if failed hashsum check three times, ignore the error to prevent an infinite loop
                print("Failed to pass hashsum check 5 times. Ignoring the error.")
                break
            # notify the user
            print("Corrupt data file detected")
            print("Expected hashsum value: "+hashsums[filename]+"\nBut calculated: "+filesum)
            os.remove(path+'/'+filename) # remove the corrupted file
            download(year,month,path) # download it again

    elif not filename in os.listdir(path): # if the file is not available, but download is turned off
        print ('Can\'t find data for {}/{}. Skipping.'.format(month, year)) # notify the user
        return

    # open the file as a text file, in utf8 encoding
    fin = bz2.BZ2File(path+'/'+filename,'r')

    # Get names of processing files
    fns=get_parser_fns(year, month, path)

    # create a file to write the processed text to
    if NN == True: # if doing NN
        fout = open(fns["nn_prep"],'w')
    else: # if doing LDA
        fout = open(fns["lda_prep"],'w')

    # create a file if we want to write the original comments and their indices to disk
    if write_original == True:
        foriginal = open(fns["original_comm"],'w')
        main_indices = open(fns["original_indices"],'w')

    # if we want to record sign of the votes
    if vote_counting == True:
        # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
        vote = open(fns["votes"], 'w')

    # create a file to store the relevant cummulative indices for each month
    ccount = open(fns["counts"], 'w')

    main_counter=0
    processed_counter=0
    ## read data
    for line in fin: # for each comment
        main_counter += 1 # update the general counter

        # parse the json, and turn it into regular text
        comment = decoder.decode(line)
        original_body = HTMLParser.HTMLParser().unescape(comment["body"])

        # filter comments by relevance to the topic
        if len(GAYMAR.findall(original_body)) > 0 or len(MAREQU.findall(original_body)) > 0:
            ## preprocess the comments
            if NN == True:
                body = sent_detector.tokenize(original_body) # tokenize the sentences
                body = NN_clean(body,stop) # clean the text for NN
                if len(body) > 0: # if the comment body is not empty after preprocessing
                    processed_counter += 1 # update the counter
                    # if we want to write the original comment to disk
                    if write_original == True:
                        original_body = original_body.replace("\n","") # remove mid-comment lines
                        original_body = original_body.encode("utf-8") # set encoding
                        print(" ".join(original_body.split()),file=foriginal) # record the original comment
                        print(main_counter,file=main_indices) # record the main index

                    for sen in body: # for each sentence in the comment
                        # remove mid-comment lines and set encoding
                        sen = sen.replace("\n","")
                        sen = sen.encode("utf-8")

                        # print the processed sentence to file
                        print(" ".join(sen.split()), end=" ", file=fout)

                    # ensure that each comment is on a separate line
                    print("\n",end="",file=fout)

            else: # if doing LDA
                body = LDA_clean(original_body,stop) # clean the text for LDA
                if body.strip() != "": # if the comment is not empty after preprocessing
                    processed_counter += 1 # update the counter

                    # if we want to write the original comment to disk
                    if write_original == True:
                        original_body = original_body.replace("\n","") # remove mid-comment lines
                        original_body = original_body.encode("utf-8") # set encoding
                        print(" ".join(original_body.split()),file=foriginal) # record the original comment
                        print(main_counter,file=main_indices) # record the index in the original files

                    # remove mid-comment lines and set encoding
                    body = body.replace("\n","")
                    body = body.encode("utf-8")

                    # print the comment to file
                    print(" ".join(body.split()), sep=" ",end="\n", file=fout)

            # if we are interested in the sign of the votes
            if vote_counting == True:
                # write the sign of the vote to file (-1 if negative, 0 if neutral, 1 if positive)
                print(int(comment["score"]),end="\n",file=vote)
 
            # record the number of documents by year and month
            created_at = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime('%Y-%m')
            timedict[created_at]=timedict.get(created_at, 0)
            timedict[created_at]+=1

    # write the monthly cummulative number of comments to file
    print(processed_counter,file=ccount)

    # close the files to save the data
    fin.close()
    fout.close()
    if NN == True:
        vote.close()
    if write_original == True:
        foriginal.close()
        main_indices.close()
    ccount.close()
    with open(fns["timedict"], "wb") as wfh:
        pickle.dump(timedict, wfh) 

    # timer
    print("Finished parsing "+filename+" at " + time.strftime('%l:%M%p'))

    if clean_raw: # if the user wishes compressed data files to be removed after processing
        print ("Cleaning up {}/{}.".format(path, filename))
        os.system('cd {} && rm {}'.format(path, filename)) # delete the recently processed file

    return

### define the parser

# NOTE: Parses for LDA if NN = False
# NOTE: Saves the text of the non-processed comment to file as well if write_original = True

### define the parser

# NOTE: Parses for LDA if NN = False
# NOTE: Saves the text of the non-processed comment to file as well if write_original = True

def pool_parsing_data(path=path, NN=NN):
    fns=get_parser_fns(path=path)
    # Initialize an "overall" timedict
    timedict=defaultdict(lambda:0)
    for kind in fns.keys():
        fns_=[ get_parser_fns(year, month, path)[kind] for year, month in dates
             ]
        if kind=="timedict":
            # Update overall timedict with data from each year
            for fn_ in fns_:
                with open(fn_, "rb") as rfh:
                    minitimedict=pickle.load(rfh)
                    for mo, val in minitimedict.items():
                        timedict[mo]+=val    
            with open(fns["timedict"], "w") as wfh:
                for month,docs in sorted(timedict.iteritems()):
                    print(month+" "+str(docs), end='\n', file=wfh)
            continue
        subprocess.call("cat "+" ".join(fns_)+"> "+fns[kind], shell=True)

def Parser(dates=dates, path=path, stop=stop, vote_counting=vote_counting,
           NN=NN, write_original=WRITE_ORIGINAL, download_raw=DOWNLOAD_RAW,
           clean_raw=CLEAN_RAW):

    # get the correct hashsums to check file integrity
    hashsums = Get_Hashsums(path)
   
    # Parallelize parsing by month 
    pool = multiprocessing.Pool(processes=CpuInfo())
    inputs=[ (year, month, hashsums, path, stop, vote_counting, NN,
              write_original, download_raw, clean_raw) for year, month in dates
           ]
    pool.map(parse_one_month_wrapper, inputs)

    # timer
    print("Finished parsing at " + time.strftime('%l:%M%p'))

    # Pool parsing data from all files
    pool_parsing_data(path, NN)

### Function to call parser when needed and parse comments

# Parameters:

#   dates: a list of (year,month) tuples for which data is to be processed
#   path: Path for data and output files.
#   stop: List of stopwords.
#   vote_counting: Include number of votes per comment in parsed file.
#   NN: Parse for neural net.
#   write_original: Write a copy of the raw file.
#   download_raw: If the raw data doesn't exist in path, download a copy from
#       https://files.pushshift.io/reddit/comments/.
#   clean_raw: Delete the raw data file when finished.

# TODO: Replace mentions of Vote in this file with mentions of sample_ratings

def Parse_Rel_RC_Comments(dates=dates, path=path, stop=stop, NN=NN,
                          vote_counting=vote_counting,
                          write_original=WRITE_ORIGINAL,
                          download_raw=DOWNLOAD_RAW, clean_raw=CLEAN_RAW):

    # check input arguments for valid type
    assert type(vote_counting) is bool
    assert type(NN) is bool
    assert type(write_original) is bool
    assert type(download_raw) is bool
    assert type(clean_raw) is bool
    assert type(path) is str
    assert type(stop) is set or type(stop) is list

    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')

    # if preprocessed comments are available, ask if they should be rewritten
    if (NN == True and Path(path+"/nn_prep").is_file()) or (NN == False and Path(path+"/lda_prep").is_file()):
        Q = raw_input("Preprocessed comments are already available. Do you wish to delete them and parse again [Y/N]?")

        if Q == 'Y' or Q == 'y': # if the user wishes to overwrite the comments

            # delete previous preprocessed data
            if NN == True: # for NN
                os.remove(path+"/nn_prep")
                if Path(path+"/votes").is_file():
                    os.remove(path+"/votes")

            elif NN == False: # for LDA
                os.remove(path+"/lda_prep")
            if Path(path+"/RC_Count_List").is_file():
                os.remove(path+"/RC_Count_List")
            if Path(path+"/RC_Count_Dict").is_file():
                os.remove(path+"/RC_Count_Dict")

            # timer
            print("Started parsing at " + time.strftime('%l:%M%p'))
            Parser(dates,path,stop,vote_counting,NN,write_original,download_raw,
                   clean_raw)

        else: # if preprocessed comments are available and the user does not wish to overwrite them

            print("Checking for missing files")

            # check for other required files aside from main data
            missing_files = 0

            if not Path(path+"/RC_Count_List").is_file():
                missing_files += 1
            if NN == True:
                if not Path(path+"/votes").is_file():
                    missing_files += 1
            # if not Path(path+"/RC_Count_Dict").is_file():
            #     missing_files += 1

            # if there are missing files, delete any partial record and parse again
            if missing_files != 0:

                print("Deleting partial record and parsing again")

                if NN == True: # for NN
                    os.remove(path+"/nn_prep")
                    if Path(path+"/votes").is_file():
                        os.remove(path+"/votes")

                elif NN == False: # for LDA
                    os.remove(path+"/lda_prep")
                if Path(path+"/RC_Count_List").is_file():
                    os.remove(path+"/RC_Count_List")
                # if Path(path+"/RC_Count_Dict").is_file():
                #     os.remove(path+"/RC_Count_Dict")

                # timer
                print("Started parsing at " + time.strftime('%l:%M%p'))
                Parser(dates,path,stop,vote_counting,NN,write_original, download_raw,
                       clean_raw)

    else:
        if Path(path+"/RC_Count_List").is_file():
            os.remove(path+"/RC_Count_List")
        if NN == True:
            if Path(path+"/votes").is_file():
                os.remove(path+"/votes")
        # if Path(path+"/RC_Count_Dict").is_file():
        #     os.remove(path+"/RC_Count_Dict")

        # timer
        print("Started parsing at " + time.strftime('%l:%M%p'))
        Parser(dates,path,stop,vote_counting,NN,write_original,download_raw,clean_raw)

### calculate the yearly relevant comment counts

def Yearly_Counts(path=path):

    # check for monthly relevant comment counts
    if not Path(path+'/RC_Count_List').is_file():
        raise Exception('The cummulative monthly counts could not be found')

    # load monthly relevant comment counts
    with open(path+"/RC_Count_List",'r') as f:
        timelist = []
        for line in f:
            if line.strip() != "":
                timelist.append(int(line))

    # calculate the cummulative yearly counts

    # intialize lists and counters
    cumm_rel_year = [] # cummulative number of comments per year
    relevant_year = [] # number of comments per year

    month_counter = 0

    # iterate through monthly counts
    for index,number in enumerate(timelist): # for each month
        month_counter += 1 # update counter

        if month_counter == 12 or index == len(timelist) - 1: # if at the end of the year or the corpus

            cumm_rel_year.append(number) # add the cummulative count

            if index + 1 == 12: # for the first year

                relevant_year.append(number) # append the cummulative value to number of comments per year

            else: # for the other years, subtract the last two cummulative values to find the number of relevant comments in that year

                relevant_year.append(number - cumm_rel_year[-2])

            month_counter = 0 # reset the counter at the end of the year

    assert sum(relevant_year) == cumm_rel_year[-1]
    assert cumm_rel_year[-1] == timelist[-1]

    return relevant_year,cumm_rel_year

### Helper functions for select_random_comments

def get_comment_lengths(path=path):
    fin=path+'/lda_prep'
    with open(fin, 'r') as fh:
        return [ len(line.split()) for line in fh.read().split("\n") ]

def _select_n(n, iterable):
    if len(iterable)<n:
        return iterable
    return np.random.choice(iterable, size=n, replace=False)

#### Writes the indices of n comments from each year in years to file.

# Parameters:
#   path: Path to working directory.
#   years_to_sample: Years to select from.
#   min_n_comments: Combine all comments from years with less than
#       min_n_comments comments and select from the combined set. E.g. If
#       min_n_comments = 5000, since there are less than 5000 (relevant)
#       comments from 2006, 2007 and 2008, a random sample of n will be drawn
#       from the pooled set of relevant comments from 2006, 2007 and 2008.
#       Defaults to 5000.
#   overwrite: If the sample file for the year exists, skip.

def select_random_comments(path=path, n=n_random_comments,
                           years_to_sample=years, min_n_comments=5000,
                           overwrite=OVERWRITE):
    # File to write random comment indices to
    fout='random_indices'
    fcounts='random_indices_count'

    if path[-1]!='/':
        path+='/'
    fout=path+fout
    fcounts=path+fcounts
    if ( not overwrite and os.path.exists(fout) ):
        print ("{} exists. Skipping. Set overwrite to True to overwrite.".format(fout))
        return

    years_to_sample=sorted(years_to_sample)
    ct_peryear, ct_cumyear=Yearly_Counts(path)
    ct_lu=dict((y, i) for i, y in enumerate(years))
    early_years=[ yr for yr in years_to_sample if
                  ct_peryear[ct_lu[yr]]<min_n_comments ]

    # Make sure the early_years actually contains the first years in years, if
    # any. Otherwise the order that indices are written to file won't make any
    # sense.

    assert all([ early_years[i]==early_years[i-1]+1 for i in range(1,
                 len(early_years)) ])
    assert all([ yr==yr_ for yr, yr_ in zip(early_years,
                 years_to_sample[:len(early_years)]) ])

    later_years=[ yr for yr in years_to_sample if yr not in early_years ]

    # Record the number of indices sampled per year
    nixs=defaultdict(int)

    # Get a list of comment lengths, so we can filter by it
    lens=get_comment_lengths(path)

    with open(fout, 'w') as wfh:
        if len(early_years)>0:
            fyear, lyear=early_years[0], early_years[-1]
            start=ct_cumyear[ct_lu[fyear-1]] if fyear-1 in ct_lu else 0
            end=ct_cumyear[ct_lu[lyear]]
            ixs_longenough=[ ix for ix in range(start, end) if lens[ix] >=
                             min_comm_length ]
            ixs=sorted(_select_n(n, ixs_longenough))
            for ix in ixs:
                nixs[years[[ ct>ix for ct in ct_cumyear ].index(True)]]+=1
            assert sum(nixs.values())==len(ixs)
            wfh.write('\n'.join(map(str, ixs)))
            wfh.write('\n')
        for year in later_years:
            start=ct_cumyear[ct_lu[year-1]]
            end=ct_cumyear[ct_lu[year]]
            ixs_longenough=[ ix for ix in range(start, end) if lens[ix] >= 
                             min_comm_length ]
            ixs=sorted(_select_n(n, ixs_longenough))
            nixs[year]=len(ixs)
            wfh.write('\n'.join(map(str, ixs)))
            wfh.write('\n')

    with open(fcounts, 'w') as wfh:
        wfh.write('\n'.join('{} {}'.format(k, v) for k, v in
                  sorted(nixs.iteritems(), key=lambda kv: kv[0])))

### determine what percentage of the posts in each year was relevant based on content filters

# NOTE: Requires total comment counts (RC_Count_Total) from http://files.pushshift.io/reddit/comments/
# NOTE: Requires monthly relevant counts from parser or disk

def Rel_Counter(path):

    # check paths
    # if not Path(path+"/RC_Count_Dict").is_file():
    #     raise Exception('Monthly counts cannot be found')
    if not Path(path+"/RC_Count_List").is_file():
        raise Exception('Cumulative monthly comment counts could not be found')
    if not Path(path+"/RC_Count_Total").is_file():
        raise Exception('Total monthly comment counts could not be found')

    # load the total monthly counts into a dictionary
    d = {}
    with open(path+"/RC_Count_Total",'r') as f:
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

    relevant_year, _ = Yearly_Counts(path)
    relevant = {}
    for idx,year in enumerate(relevant_year):
        relevant[str(2006+idx)] = year

    # calculate the percentage of comments in each year that was relevant and write it to file
    perc_rel = {}
    rel = open(path+"/perc_rel",'a+')
    for key in relevant:
        perc_rel[key] = float(relevant[key]) / float(total_year[key])
    print(sorted(perc_rel.items()),file=rel)
    rel.close

### Load, calculate or re-calculate the percentage of relevant comments/year

def Perc_Rel_RC_Comment(path):

    if Path(path+"/perc_rel").is_file(): # look for extant record

        # if it exists, ask if it should be overwritten
        Q = raw_input("Yearly relevant percentages are already available. Do you wish to delete them and count again [Y/N]?")

        if Q == 'Y' or Q == 'y': # if yes

            os.remove(path+"/perc_rel") # delete previous record
            Rel_Counter(path) # calculate again

        else: # if no

            print("Operation aborted") # pass

    else: # if there is not previous record

        Rel_Counter(path) # calculate

### function to determine comment indices for new training, development and test sets

def Create_New_Sets(regression,path,output_path,training_fraction,indices,NN,all_):

    print("Creating sets")

    # determine number of comments in the dataset
    if all_: # if doing NN or processing the entire corpus for LDA

        if regression==False: # if not doing regression on sampled comments

            num_comm = indices[-1] # retrieve the total number of comments
            indices = range(num_comm) # define sets over all comments

        else: # if doing regression on sampled comments

            # check to see if human comment ratings can be found on disk
            if not Path(output_path+'/sample_ratings.csv').is_file():
                raise Exception("Human comment ratings for regressor training could not be found on file.")

            # retrieve the number of comments for which there are complete human ratings
            with open(output_path+'/sample_ratings.csv','r+b') as csvfile:
                reader = csv.reader(csvfile)
                human_ratings = [] # initialize counter for the number of valid human ratings
                # read human data for sampled comments one by one
                for idx,row in enumerate(reader):
                    row = row[0].split(",")
                    # ignore headers and record the index of comments that are interpretable and that have ratings for all three goal variables
                    if idx != 0 and (row[7] != 'N' or row[7] != 'n') and is_number(row[4]) and is_number(row[5]) and is_number(row[6]):
                        human_ratings.append(int(row[1]))

            num_comm = len(human_ratings) # the number of valid samples for network training
            indices = human_ratings # define sets over sampled comments with human ratings

    else: # if using LDA on a random subsample of the comments
        num_comm = len(indices) # total number of sampled comments
        # in this case, the input indices do comprise the set we're looking for

    num_train = int(ceil(training_fraction * num_comm)) # size of training set

    if NN == True: # for NN

        num_remaining = num_comm - num_train # the number of comments in development set or test set
        num_dev = int(floor(num_remaining/2)) # size of the development set
        num_test = num_remaining - num_dev # size of the test set

        sets['dev'] = sample(indices, num_dev) # choose development comments at random
        remaining = set(indices).difference(sets['dev'])
        sets['test']  = sample(remaining,num_test) # choose test comments at random
        # use the rest as training set
        sets['train'] = set(remaining).difference(sets['test'])

        # sort the indices based on position in nn_prep
        for set_key in set_key_list:
            sets[set_key] = sorted(list(sets[set_key]))

        # Check dev and test sets came out with right proportions
        assert (len(sets['dev']) - len(sets['test'])) <= 1
        assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == len(indices)

        # write the sets to file
        for set_key in set_key_list:
            with open(path+'/'+set_key+'_set_'+str(regression),'a+') as f:
                for index in sets[set_key]:
                    print(index,end='\n',file=f)

    else: # for LDA

        num_eval = num_comm - num_train # size of evaluation set

        LDA_sets['eval'] = sample(indices,num_eval) # choose evaluation comments at random
        LDA_sets['train'] = set(indices).difference(set(LDA_sets['eval'])) # assign the rest of the comments to training

        # sort the indices based on position in lda_prep
        for set_key in LDA_set_keys:
            LDA_sets[set_key] = sorted(list(LDA_sets[set_key]))

        # Check that sets came out with right proportions
        assert len(LDA_sets['train']) + len(LDA_sets['eval']) == len(indices)

        # write the sets to file
        for set_key in LDA_set_keys:
            with open(path+'/LDA_'+set_key+'_set_'+str(all_),'a+') as f:
                for index in LDA_sets[set_key]:
                    print(index,end='\n',file=f)

### function for loading, calculating, or recalculating sets

# helper function for checking if a string is a number

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def Define_Sets(regression,path=path, training_fraction=training_fraction, NN=NN,
                all_=ENTIRE_CORPUS):

    ## ensure the arguments have the correct types and values

    assert type(path) is str
    assert 0 < training_fraction and 1 > training_fraction
    assert type(NN) is bool

    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')

    # load the number of comments or raise Exception if they can't be found
    findices='RC_Count_List' if all_ else 'random_indices'
    try:
        assert findices in os.listdir(path)
    except AssertionError:
        raise Exception("File {} not found.".format(findices))

    indices=open(path+'/'+findices, 'r').read().split()
    indices=filter(lambda x:x.strip(), indices)
    indices=map(int, indices)

    # if indexed comments are available (NN)
    if (NN == True and Path(path+"/indexed_train_"+str(regression)).is_file() and Path(path+"/indexed_dev_"+str(regression)).is_file() and Path(path+"/indexed_test_"+str(regression)).is_file()):

        # determine if the comments and their relevant indices should be deleted and re-initialized or the sets should just be loaded
        Q = raw_input("Indexed comments are already available. Do you wish to delete sets and create new ones [Y/N]?")

        # If recreating the sets is requested, delete the current ones and reinitialize

        if Q == "Y" or Q == "y":

            print("Deleting any existing sets and indexed comments")

            # delete previous record
            for set_key in set_key_list:
                if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():
                    os.remove(path+"/indexed_"+set_key+"_"+str(regression))
                if Path(path+"/"+set_key+"_set_"+str(regression)).is_file():
                    os.remove(path+"/"+set_key+"_set_"+str(regression))

            Create_New_Sets(regression,path,output_path,training_fraction,indices, NN, all_) # create sets

        # If recreating is not requested, attempt to load the sets
        elif Q == "N" or Q == "n":

            # if the sets are found, load them
            if Path(path+"/train_set_"+str(regression)).is_file() and Path(path+"/dev_set_"+str(regression)).is_file() and Path(path+"/test_set_"+str(regression)).is_file():

                print("Loading sets from file")

                for set_key in set_key_list:
                    with open(path+'/'+set_key + '_set_' + str(regression),'r') as f:
                        for line in f:
                            if line.strip() != "":
                                sets[set_key].append(int(line))
                    sets[set_key] = np.asarray(sets[set_key])

                # ensure set sizes are correct
                assert len(sets['dev']) - len(sets['test']) < 1
                assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == len(indices)

            else: # if the sets cannot be found, delete any current sets and create new sets

                print("Failed to load previous sets. Reinitializing")

                # delete partial record
                for set_key in set_key_list:
                    if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():
                        os.remove(path+"/indexed_"+set_key+"_"+str(regression))
                    if Path(path+"/"+set_key+"_set").is_file():
                        os.remove(path+"/"+set_key+"_set_"+str(regression))

                Create_New_Sets(regression,path,output_path,training_fraction,indices, NN, all_) # create sets

        else: # if response was something other tha Y or N
            print("Operation aborted")
            pass

    else: # no indexed comments available or not creating sets for NN

        # delete any possible partial indexed set
        if NN == True:
            for set_key in set_key_list:
                if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():
                    os.remove(path+"/indexed_"+set_key+"_"+str(regression))

        # check to see if there are sets available, if so load them
        if (NN == True and Path(path+"/train_set_"+str(regression)).is_file() and Path(path+"/dev_set_"+str(regression)).is_file() and Path(path+"/test_set_"+str(regression)).is_file()) or (NN == False and Path(path+"/LDA_train_set_"+str(all_)).is_file() and Path(path+"/LDA_eval_set_"+str(all_)).is_file()):

            print("Loading sets from file")

            if NN == True: # for NN
                for set_key in set_key_list:
                    with open(path+'/'+set_key + '_set_'+str(regression),'r') as f:
                        for line in f:
                            if line.strip() != "":
                                sets[set_key].append(int(line))
                    sets[set_key] = np.asarray(sets[set_key])

                # ensure set sizes are correct
                assert len(sets['dev']) - len(sets['test']) < 1
                assert len(sets['dev']) + len(sets['test']) + len(sets['train']) == len(indices)

            else: # for LDA

                for set_key in LDA_set_keys:
                    with open(path+"/LDA_"+set_key+"_set_"+str(all_),'r') as f:
                        for line in f:
                            if line.strip() != "":
                                LDA_sets[set_key].append(int(line))
                    LDA_sets[set_key] = np.asarray(LDA_sets[set_key])

        else: # if not all sets are found

            if NN == True: # for NN

                # delete any partial set
                for set_key in set_key_list:
                    if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():
                        os.remove(path+"/indexed_"+set_key+"_"+str(regression))
                    if Path(path+"/"+set_key+"_set_"+str(regression)).is_file():
                        os.remove(path+"/"+set_key+"_set_"+str(regression))

                # create new sets
                Create_New_Sets(regression,path,output_path,training_fraction,indices,NN, all_)

            else: # for LDA

                # delete any partial set
                for set_key in LDA_set_keys:
                    if Path(path+"/LDA_"+set_key+"_set_"+str(all_)).is_file():
                        os.remove(path+"/LDA_"+set_key+"_set_"+str(all_))

                # create new sets
                Create_New_Sets(regression,path,output_path,training_fraction,indices,NN,all_)

### load or create vocabulary and load or create indexed versions of comments in sets

# NOTE: Only for NN. For LDA we use gensim's dictionary functions

def Index_Set(regression,path,set_key,MaxVocab,FrequencyFilter):

    # ensure the arguments have the correct type
    assert type(path) is str
    assert type(MaxVocab) is int
    assert type(FrequencyFilter) is int

    # check the given path
    if not os.path.exists(path):
        raise Exception('Invalid path')

    ## record word frequency in the entire dataset
    frequency = defaultdict(int)

    if Path(path+"/nn_prep").is_file(): # look for preprocessed data

        fin = open(path+'/nn_prep','r')
        for comment in fin: # for each comment
            for token in comment.split(): # for each word
                frequency[token] += 1 # count the number of occurrences

    else: # if no data is found, raise an error
        raise Exception('Pre-processed dataset could not be found')

    # if indexed comments are available and we are trying to index the training set
    if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file() and set_key == 'train':

        # If the vocabulary is available, load it
        if Path(path+"/dict_"+str(regression)).is_file():
            print("Loading dictionary from file")

            with open(path+"/dict_"+str(regression),'r') as f:
                for line in f:
                    if line.strip() != "":
                        (key, val) = line.split()
                        V[key] = int(val)

        else: # if the vocabulary is not available

            # delete the possible dictionary-less indexed training set file
            if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():
                os.remove(path+"/indexed_"+set_key+"_"+str(regression))

    # if indexed comments are available, load them
    if Path(path+"/indexed_"+set_key+"_"+str(regression)).is_file():

        print("Loading the set from file")

        with open(path+"/indexed_"+set_key+"_"+str(regression),'r') as f:
            for line in f:
                assert line.strip() != ""
                comment = []
                for index in line.split():
                    comment.append(index)
                indexes[set_key].append(comment)

    else: # if the indexed comments are not available, create them

        if set_key == 'train': # for training set

            # timer
            print("Started creating the dictionary at " + time.strftime('%l:%M%p'))

            ## initialize the vocabulary with various UNKs

            V.update({"*STOP2*":1,"*UNK*":2,"*UNKED*":3,"*UNKS*":4,"*UNKING*":5,"*UNKLY*":6,"*UNKER*":7,"*UNKION*":8,"*UNKAL*":9,"*UNKOUS*":10,"*STOP*":11})

        ## read the dataset and index the relevant comments

        fin.seek(0) # go to the beginning of the data file
        for counter,comm in enumerate(fin): # for each comment

            if counter in sets[set_key]: # if it belongs in the set

                comment = [] # initialize a list

                for word in comm.split(): # for each word

                    if frequency[word] > FrequencyFilter: # filter non-frequent words

                        if word in V.keys(): # if the word is already in the vocabulary
                            comment.append(V[word]) # index it and add it to the list

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

                indexes[set_key].append(comment) # add the comment to the indexed list

        ## save the vocabulary to file

        if set_key == 'train':
            vocab = open(path+"/dict_"+str(regression),'a+')
            for word,index in V.iteritems():
                print(word+" "+str(index),file=vocab)
            vocab.close

        ## save the indexed datasets to file

        with open(path+"/indexed_"+set_key+"_"+str(regression),'a+') as f:
            for comment in indexes[set_key]:
                assert len(comment) != 0
                for ind,word in enumerate(comment):
                    if ind != len(comment) - 1:
                        print(word,end=" ",file=f)
                    elif ind == len(comment) - 1:
                        print(word,file=f)

        # ensure that datasets have the right size
        assert len(indexes[set_key]) == len(sets[set_key])

    # timer
    print("Finished indexing the "+set_key+" set at " + time.strftime('%l:%M%p'))

# ### turn votes into one-hot vectors (only for classifier NN - Not language model or regression)
#
# def One_Hot_Vote(vote_list):
#
#     one_hot_vote = []
#
#     for sign in vote_list:
#         if int(sign) == -1:
#             one_hot_vote.append([1,0,0])
#         elif int(sign) == 0:
#             one_hot_vote.append([0,1,0])
#         elif int(sign) == 1:
#             one_hot_vote.append([0,0,1])
#
#         else:
#             raise Exception('Votes could not be appended')
#
#     return one_hot_vote
#
# ### import the correct labels for comment votes (only for classifier NN)
#
# def Get_Votes(path):
#
#     # look for data on disk
#     if Path(path+"/votes").is_file():
#
#         # load the votes or raise an error if a vote is not assigned to a set
#         for set_key in set_key_list:
#             vote[set_key] = []
#
#         with open(path+"/votes",'r') as f:
#             for index,sign in enumerate(f):
#                 sign = sign.strip()
#                 match_found = 0
#                 for set_key in set_key_list:
#                     if index in sets[set_key]:
#                         vote[set_key].append(sign)
#                         match_found += 1
#
#                 if match_found == 0:
#                     raise Exception('Votes could not be read from file')
#
#         # ensure the datasets have the right sizes
#         assert len(indexes['train']) == len(vote['train'])
#         assert len(indexes['dev']) == len(vote['dev'])
#         assert len(indexes['test']) == len(vote['test'])
#
#     else: # if votes cannot be found on file
#         raise Exception('Labels for the sets could not be found')
#
#     # turn the votes into one-hot vectors
#     for set_key in set_key_list:
#         vote[set_key] = One_Hot_Vote(vote[set_key])
#     return vote['train'],vote['dev'],vote['test']

### Function for reading and indexing a pre-processed corpus for LDA

def LDA_Corpus_Processing(path=path, no_below=no_below, no_above=no_above,
                          MaxVocab=MaxVocab,all_=ENTIRE_CORPUS):

    # check the existence of pre-processed data and sets
    if not Path(path+'/lda_prep').is_file():
        raise Exception('Pre-processed data could not be found')
    if not Path(path+'/LDA_train_set_'+str(all_)).is_file() or not Path(path+'/LDA_eval_set_'+str(all_)).is_file():
        raise Exception('Comment sets could not be found')

    # open the file storing pre-processed comments
    f = open(path+'/lda_prep','r')

    # check to see if the corpus has previously been processed
    required_files = ['RC_LDA_Corpus_'+str(all_)+'.mm','RC_LDA_Eval_'+str(all_)+'.mm','RC_LDA_Dict_'+str(all_)+'.dict','train_word_count_'+str(all_),'eval_word_count_'+str(all_)]
    missing_file = 0
    for saved_file in required_files:
        if not Path(path+'/'+saved_file).is_file():
            missing_file += 1

    # if there is a complete extant record, load it
    if missing_file == 0:
        corpus = gensim.corpora.MmCorpus(path+'/RC_LDA_Corpus_'+str(all_)+'.mm')
        eval_comments = gensim.corpora.MmCorpus(path+'/RC_LDA_Eval_'+str(all_)+'.mm')
        dictionary = gensim.corpora.Dictionary.load(path+'/RC_LDA_Dict_'+str(all_)+'.dict')
        with open(path+'/train_word_count_'+str(all_)) as g:
            for line in g:
                if line.strip() != "":
                    train_word_count = int(line)
        with open(path+'/eval_word_count_'+str(all_)) as h:
            for line in h:
                if line.strip() != "":
                    eval_word_count = int(line)

        print("Finished loading the dictionary and the indexed corpora from file")

    # delete any incomplete corpus record
    elif missing_file > 0 and missing_file != len(required_files):
        for saved_file in required_files:
            if Path(path+'/'+saved_file).is_file():
                os.remove(path+'/'+saved_file)
        missing_file = len(required_files)

    # if there are no saved corpus files
    if missing_file == len(required_files):

        # timer
        print("Started processing the dataset at " + time.strftime('%l:%M%p'))

        f.seek(0) # go to the beginning of the file

        # initialize a list for the corpus
        texts = []
        eval_comments = []

        train_word_count = 0 # total number of words in the filtered corpus
        eval_word_count = 0 # total number of words in the evaluation set

        ## iterate through the dataset

        for index,comment in enumerate(f): # for each comment

            if index in LDA_sets['train']: # if it belongs in the training set

                document = [] # initialize a bag of words
                if len(comment.strip().split()) == 1:
                    document.append(comment.strip())
                else:
                    for word in comment.strip().split(): # for each word
                        document.append(word)

                train_word_count += len(document)
                texts.append(document) # add the BOW to the corpus

            elif index in LDA_sets['eval']: # if in evaluation set

                document = [] # initialize a bag of words
                if len(comment.strip().split()) == 1:
                    document.append(comment.strip())
                else:
                    for word in comment.strip().split(): # for each word
                        document.append(word)

                eval_word_count += len(document)
                eval_comments.append(document) # add the BOW to the corpus

            else: # if the index is in neither set and we're processing the entire corpus, raise an Exception
                if all_:
                    raise Exception('Error in processing comment indices')
                continue

        # write the number of words in the frequency-filtered corpus to file
        with open(path+'/train_word_count_'+str(all_),'w') as u:
            print(train_word_count,file=u)

        # write the number of words in the frequency-filtered evaluation set to file
        with open(path+'/eval_word_count_'+str(all_),'w') as w:
            print(eval_word_count,file=w)

        ## create the dictionary

        dictionary = gensim.corpora.Dictionary(texts,prune_at=MaxVocab) # training set
        dictionary.add_documents(eval_comments,prune_at=MaxVocab) # add evaluation set
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=MaxVocab) # filter extremes
        dictionary.save(path+'/RC_LDA_Dict_'+str(all_)+'.dict') # save dictionary to file for future use

        ## create the Bag of Words (BOW) datasets

        corpus = [dictionary.doc2bow(text) for text in texts] # turn training comments into BOWs
        eval_comments = [dictionary.doc2bow(text) for text in eval_comments] # turn evaluation comments into BOWs
        gensim.corpora.MmCorpus.serialize(path+'/RC_LDA_Corpus_'+str(all_)+'.mm', corpus) # save indexed data to file for future use (overwrites any previous versions)
        gensim.corpora.MmCorpus.serialize(path+'/RC_LDA_Eval_'+str(all_)+'.mm', eval_comments) # save the evaluation set to file

        # timer
        print("Finished creating the dictionary and the term-document matrices at "+time.strftime('%l:%M%p'))

    return dictionary, corpus, eval_comments, train_word_count, eval_word_count

### Get lower bounds on per-word perplexity for training and development sets (LDA)

def Get_Perplexity(ldamodel,corpus,eval_comments,training_fraction,train_word_count,eval_word_count):

    # timer
    print("Started calculating perplexity at "+time.strftime('%l:%M%p'))

    ## calculate model perplexity for training and evaluation sets

    train_perplexity = ldamodel.bound(corpus, subsample_ratio = training_fraction)
    eval_perplexity = ldamodel.bound(eval_comments, subsample_ratio = 1-training_fraction)

    ## calculate per-word perplexity for training and evaluation sets

    train_per_word_perplex = np.exp2(-train_perplexity / train_word_count)
    eval_per_word_perplex = np.exp2(-eval_perplexity / eval_word_count)

    # timer
    print("Finished calculating perplexity at "+time.strftime('%l:%M%p'))

    ## Print and save the per-word perplexity values to file

    with open(output_path+"/Performance",'a+') as perf:
        print("*** Perplexity ***",file=perf)
        print("Lower bound on per-word perplexity (using "+str(training_fraction)+" percent of documents as training set): "+str(train_per_word_perplex))
        print("Lower bound on per-word perplexity (using "+str(training_fraction)+" percent of documents as training set): "+str(train_per_word_perplex),file=perf)
        print("Lower bound on per-word perplexity (using "+str(1-training_fraction)+" percent of held-out documents as evaluation set): "+str(eval_per_word_perplex))
        print("Lower bound on per-word perplexity (using "+str(1-training_fraction)+" percent of held-out documents as evaluation set): "+str(eval_per_word_perplex),file=perf)

    return train_per_word_perplex,eval_per_word_perplex

### function for creating an enhanced version of the dataset with year and comment indices (used in topic contribution and theta calculation)

# NOTE: This function will in the future be removed and integrated into the main parser

def Get_Indexed_Dataset(path,cumm_rel_year,all_=ENTIRE_CORPUS):

    with open(path+'/lda_prep','r') as f:

        indexed_dataset = [] # initialize the full dataset

        year_counter = 0 # the first year in the corpus (2006)

        if not all_:
            assert Path(path+'/random_indices').is_file()
            with open(path+'/random_indices') as g:
                rand_subsample = []
                for line in g:
                    if line.strip() != "":
                        rand_subsample.append(int(line))

        for comm_index,comment in enumerate(f): # for each comment

            if comm_index >= cumm_rel_year[year_counter]:
                year_counter += 1 # update the year counter if need be

            if all_ or (not all_ and comm_index in rand_subsample):
                indexed_dataset.append((comm_index,comment,year_counter)) # append the comment and the relevant year to the dataset

    return indexed_dataset

### Topic Contribution (serial) ###

### Function that goes through the corpus and calculates the contribution of each topic to comment content in each year

# NOTE: no_above, no_below and MaxVocab filters mean that you might encounter a word in the dataset that is not in the dictionary

# def Topic_Contribution(path,dictionary,ldamodel,relevant_year,cumm_rel_year,num_topics):
#
#     # timer
#     print("Started calculating topic contribution at " + time.strftime('%l:%M%p'))
#
#     # initialize a vector for yearly topic contribution
#     yr_topic_cont_serial = np.zeros([len(cumm_rel_year),num_topics])
#
#     if not Path(path+'/lda_prep').is_file():
#         raise Exception('The preprocessed data could not be found')
#
#     with open(path+'/lda_prep','r') as f:
#         # iterate through the corpus and calculate topic contribution
#         year_counter = 0 # the first year in the corpus (2006)
#         for comm_index,comment in enumerate(f): # for each comment in the dataset
#
#             # update the year currently being processed based on the document index if necessary
#             if comm_index >= cumm_rel_year[year_counter]:
#                 year_counter += 1
#
#             # initialize a vector for document-topic assignments
#             dxt = np.zeros([1,num_topics])
#
#             # for each word in the document
#             analyzed_comment_length = 0 # processed comment length counter
#
#             if len(comment.strip().split()) == 1: # if comment only consists of one word after preprocessing
#                 if comment.strip() in dictionary.values(): # if word is in the dictionary (so that predictions can be derived for it)
#                     term_topics = ldamodel.get_term_topics(dictionary.token2id[comment.strip()]) # get topic distribution for the word based on trained model
#                     if len(term_topics) != 0: # if a topic with non-trivial probability is found
#                         topic_asgmt = term_topics[np.argmax(zip(*term_topics)[1])][0] # find the most likely topic for that word
#                         dxt[0,topic_asgmt-1] += 1 # record the topic assignment
#                         analyzed_comment_length += 1 # update comment word counter
#
#             else: # if comment consists of more than one word
                # topics = ldamodel.get_document_topics(dictionary.doc2bow(indexed_comment[1].strip().split()),per_word_topics=True) # get per-word topic probabilities for the document
                # for wt_tuple in topics[1]: # iterate over the word-topic assignments
                #
                #     if len(wt_tuple[1]) != 0: # if the model has predictions for the specific word
                #
                #         # record the most likely topic according to the trained model
                #         dxt[wt_tuple[1][0],0] += 1
                #         analyzed_comment_length += 1 # update word counter
#
#             if analyzed_comment_length > 0: # if the model had predictions for at least some of the words in the comment
#                 dxt = (float(1) / float(len(comment))) * dxt # normalize the topic contribution using comment length (should it be analyzed_comment_length or length?)
#                 yr_topic_cont_serial[year_counter,:] = yr_topic_cont_serial[year_counter,:] + dxt # add topic contributions to this comment to the running yearly sum
#
#     # normalize contributions using the number of documents per year
#
#     for i in range(len(cumm_rel_year)): # for each year
#         yr_topic_cont_serial[i,:] = ( float(1) / float(relevant_year[i]) ) * yr_topic_cont_serial[i,:]
#     return yr_topic_cont_serial

### Topic Contribution (threaded) ###

### prints information about multiprocessing or threading in the current task

def info(title):

    print(title)

    print('module name:', __name__)

    if hasattr(os, 'getppid'):  # only available on Unix

        print('parent process:', os.getppid())

    print('process id:', os.getpid())

# ### function for retrieving topic distributions of a certain document
#
# def Topic_Asgmt_Retriever_Threaded(name,indexed_dataset,dictionary,ldamodel,num_topics):
#
#     for indexed_comment in indexed_dataset:
#
#         # info('function Topic_Asgmt_Retriever') # uncomment if you wish to observe which worker is processing a specific comment
#
#         # initialize a vector for document-topic assignments
#         dxt = np.zeros([1,num_topics])
#         # for each word in the document
#         analyzed_comment_length = 0 # processed comment length counter
#
#         if len(indexed_comment[1].strip().split()) == 1: # if comment only consists of one word after preprocessing
#             if indexed_comment[1].strip() in dictionary.values(): # if word is in the dictionary (so that predictions can be derived for it)
#                 term_topics = ldamodel.get_term_topics(dictionary.token2id[indexed_comment[1].strip()]) # get topic distribution for the word based on trained model
#                 if len(term_topics) != 0: # if a topic with non-trivial probability is found
#                     topic_asgmt = term_topics[np.argmax(zip(*term_topics)[1])][0] # find the most likely topic for that word
#                     dxt[0,topic_asgmt] += 1 # record the topic assignment
#                     analyzed_comment_length += 1 # update comment word counter
#
#         else: # if comment consists of more than one word
            # topics = ldamodel.get_document_topics(dictionary.doc2bow(indexed_comment[1].strip().split()),per_word_topics=True) # get per-word topic probabilities for the document

            # for wt_tuple in topics[1]: # iterate over the word-topic assignments
            #
            #     if len(wt_tuple[1]) != 0: # if the model has predictions for the specific word
            #
            #         # record the most likely topic according to the trained model
            #         dxt[wt_tuple[1][0],0] += 1
            #         analyzed_comment_length += 1 # update word counter
#
#         lock.acquire()
#         if analyzed_comment_length > 0: # if the model had predictions for at least some of the words in the comment
#             dxt = (float(1) / float(analyzed_comment_length)) * dxt # normalize the topic contribution using comment length (should it be analyzed_comment_length or length?)
#             global yr_topic_cont_threaded
#             yr_topic_cont_threaded[indexed_comment[2],:] = yr_topic_cont_threaded[indexed_comment[2],:] + dxt # add topic contributions to this comment to the running yearly sum
#         lock.release()
#
# ## main function
#
# lock=Lock()
#
# def chunkIt(seq, num):
#     avg = len(seq) / float(num)
#     out = []
#     last = 0.0
#
#     while last < len(seq):
#         out.append(seq[int(last):int(last + avg)])
#         last += avg
#
#     return out
#
# def Topic_Contribution_Threaded(path,output_path,dictionary,ldamodel,relevant_year,cumm_rel_year,num_topics,num_threads):
#
#     # timer
#     print("Started calculating topic contribution at " + time.strftime('%l:%M%p'))
#
#     # initialize a vector for yearly topic contribution
#
#     global yr_topic_cont_threaded
#     yr_topic_cont_threaded = np.zeros([len(cumm_rel_year),num_topics])
#
#     if not Path(path+'/lda_prep').is_file():
#         raise Exception('The preprocessed data could not be found')
#
#     # read and index comments
#     with open(path+'/lda_prep','r') as f:
#
#         indexed_dataset = [] # initialize the full dataset
#
#         year_counter = 0 # the first year in the corpus (2006)
#         for comm_index,comment in enumerate(f):
#             if comm_index >= cumm_rel_year[year_counter]:
#                 year_counter += 1
#             indexed_dataset.append((comm_index,comment,year_counter))
#
#     thread_sets = chunkIt(range(len(indexed_dataset)),num_threads)
#
#     thread_comments = {}
#     for i in range(num_threads):
#         thread_comments[i] = [comment for comment in indexed_dataset if comment[0] in thread_sets[i]]
#
#     threads = {}
#     for i in range(num_threads):
#         threads[i] = Thread( target=Topic_Asgmt_Retriever_Threaded, args=("Thread-"+str(i),thread_comments[i],dictionary,ldamodel,num_topics ) )
#         threads[i].start()
#         threads[i].join()
#
#     # normalize contributions using the number of documents per year
#     for i in range(len(cumm_rel_year)): # for each year
#         yr_topic_cont_threaded[i,:] = ( float(1) / float(relevant_year[i]) ) * yr_topic_cont_threaded[i,:]
#
#     np.savetxt(output_path+"/yr_topic_cont_threaded", yr_topic_cont_threaded) # save the topic contribution matrix to file
#
#     # timer
#     print("Finished calculating topic contributions at "+time.strftime('%l:%M%p'))
#
#     return yr_topic_cont_threaded

## Topic Contribution (Multicore) ###

# NOTE: Uses global vectors shared between processes. Slower than the version that comes next

## Define a class of vectors in basic C that will be shared between multi-core prcoesses for calculating topic contribution

class Shared_Contribution_Array(object):

    ## Shared_Contribution_Array attributes

    def __init__(self,num_topics):
        self.val = multiprocessing.RawArray('f', np.zeros([num_topics,1])) # shape and data type
        self.lock = multiprocessing.Lock() # prevents different processes from writing the shared variables at the same time and mixing data up

    ## Shared_Contribution_Array update method

    def Update_Val(self,dxt):
        with self.lock: # apply the lock
            for ind,_ in enumerate(self.val[:]): # for each topic
                if dxt[ind,0] != 0: # if it was the most likely for some word in the input comment
                    self.val[ind] += dxt[ind,0] # add it's contribution to the yearly running sum

### Define a counter shared between multi-core processes

class Shared_Counter(object):

    ## Shared_Counter attributes

    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    ## Shared_Counter incrementation method

    def Increment(self):
        with self.lock:
            self.val.value += 1

    ## Property for calling the value of the shared counter

    @property
    def value(self):
        return self.val.value

### Define a function that retrieves the most likely topic for each word in a comment and calculates

def Topic_Asgmt_Retriever_Multi(indexed_comment,dictionary,ldamodel,num_topics):

    # info('function Topic_Asgmt_Retriever') ## uncomment if you wish to observe which worker is processing a specific comment

    ## initialize needed vectors

    dxt = np.zeros([num_topics,1]) # a vector for the normalized contribution of each topic to the comment
    analyzed_comment_length = 0 # a counter for the number of words in a comment for which the model has predictions

    ## for each word in the comment:

    if len(indexed_comment[1].strip().split()) == 1: # if comment only consists of one word after preprocessing
        if indexed_comment[1].strip() in dictionary.values(): # if word is in the dictionary (so that predictions can be derived for it)
            term_topics = ldamodel.get_term_topics(dictionary.token2id[indexed_comment[1].strip()]) # get topic distribution for the word based on trained model
            if len(term_topics) != 0: # if a topic with non-trivial probability is found
                # find the most likely topic for that word according to the trained model
                topic_asgmt = term_topics[np.argmax(zip(*term_topics)[1])][0]

                dxt[topic_asgmt,0] += 1 # record the topic assignment
                analyzed_comment_length += 1 # update word counter

    else: # if comment consists of more than one word

        topics = ldamodel.get_document_topics(dictionary.doc2bow(indexed_comment[1].strip().split()),per_word_topics=True) # get per-word topic probabilities for the document

        for wt_tuple in topics[1]: # iterate over the word-topic assignments

            if len(wt_tuple[1]) != 0: # if the model has predictions for the specific word

                # record the most likely topic according to the trained model
                dxt[wt_tuple[1][0],0] += 1
                analyzed_comment_length += 1 # update word counter

    if analyzed_comment_length > 0: # if the model had predictions for at least some of the words in the comment

        dxt = (float(1) / float(analyzed_comment_length)) * dxt # normalize the topic contribution using comment length

        Yearly_Running_Sums[indexed_comment[2]].Update_Val(dxt) # update the vector of yearly topic contributions

    else: # if the model had no reasonable topic proposal for any of the words in the comment

        no_predictions[indexed_comment[2]].Increment() # update the no_predictions counter

### Define the main function for multi-core calculation of topic contributions

def Topic_Contribution_Multicore(path,output_path,dictionary,ldamodel,
                    relevant_year,cumm_rel_year,num_topics,all_=ENTIRE_CORPUS):

    # timer
    print("Started calculating topic contribution at " + time.strftime('%l:%M%p'))

    ## check for the existence of the preprocessed dataset
    if not Path(path+'/lda_prep').is_file():
        raise Exception('The preprocessed data could not be found')

    ## load yearly counts for randomly sampled comments if needed

    # check for access to counts
    if not all_ and not Path(path+'/random_indices_count').is_file():
        raise Exception('The year by year counts for randomly sampled comments could not be found')
    # load counts
    if not all_:
        with open(path+'/random_indices_count') as f:
            random_counts = []
            for line in f:
                line = line.replace("\n","")
                if line.strip() != "":
                    (key, val) = line.split()
                    random_counts.append(val)

    ## initialize shared vectors for yearly topic contributions

    global Yearly_Running_Sums
    Yearly_Running_Sums = {}
    no_years = len(cumm_rel_year) if all_ else len(random_counts)

    ## Create shared counters for comments for which the model has no reasonable prediction whatsoever

    global no_predictions
    no_predictions = {}

    for i in range(no_years):
        Yearly_Running_Sums[i] = Shared_Contribution_Array(num_topics)
        no_predictions[i] = Shared_Counter(initval=0)

    ## read and index comments

    indexed_dataset = Get_Indexed_Dataset(path,cumm_rel_year)

    ## define the function for spawning processes to perform the calculations in parallel

    def testfunc(indexed_dataset,dictionary,ldamodel,num_topics):
        pool = multiprocessing.Pool(processes=CpuInfo())
        func = partial(Topic_Asgmt_Retriever_Multi, dictionary=dictionary,ldamodel=ldamodel,num_topics=num_topics)
        pool.map(func=func,iterable=indexed_dataset)
        pool.close()
        pool.join()

    ## call the multiprocessing function on the dataset

    testfunc(indexed_dataset,dictionary,ldamodel,num_topics)

    ## Gather yearly topic contribution estimates in one matrix

    yearly_output = []
    for i in range(no_years):
        yearly_output.append(Yearly_Running_Sums[i].val[:])

    yearly_output = np.asarray(yearly_output)

    ## normalize contributions using the number of documents per year
    if all_: # if processing all comments
        for i in range(no_years): # for each year
            yearly_output[i,:] = ( float(1) / (float(relevant_year[i]) - no_predictions[i].value )) * yearly_output[i,:]
    else: # if processing a random subsample
        for i in range(no_years):
            yearly_output[i,:] = ( float(1) / (float(random_counts[i]) - no_predictions[i].value )) * yearly_output[i,:]

    np.savetxt(output_path+"/yr_topic_cont", yearly_output) # save the topic contribution matrix to file

    # timer
    print("Finished calculating topic contributions at "+time.strftime('%l:%M%p'))

    return yearly_output,indexed_dataset

# ### Function for Calculating Topic Contribution ### (no shared variable between processes. Compare performance)
#
# # NOTE: Topics in Gensim use Python indexing (indices start at 0)
#
# def Topic_Asgmt_Retriever_Multi(indexed_comment,dictionary,ldamodel,num_topics):
#
#     # info('function Topic_Asgmt_Retriever') ## uncomment if you wish to observe which worker is processing a specific comment
#
#     ## initialize needed vectors
#
#     dxt = np.zeros([num_topics,1]) # a vector for the normalized contribution of each topic to the comment
#     analyzed_comment_length = 0 # a counter for the number of words in a comment for which the model has predictions
#     no_predictions = 0 # initialize a binary variable for recording whether the model has predictions for a certain comment
#
#     ## for each word in the comment:
#
#     if len(indexed_comment[1].strip().split()) == 1: # if comment only consists of one word after preprocessing
#         if indexed_comment[1].strip() in dictionary.values(): # if word is in the dictionary (so that predictions can be derived for it)
#             term_topics = ldamodel.get_term_topics(dictionary.token2id[indexed_comment[1].strip()]) # get topic distribution for the word based on trained model
#             if len(term_topics) != 0: # if a topic with non-trivial probability is found
#                 # find the most likely topic for that word according to the trained model
#                 topic_asgmt = term_topics[np.argmax(zip(*term_topics)[1])][0]
#
#                 dxt[topic_asgmt,0] += 1 # record the topic assignment
#                 analyzed_comment_length += 1 # update word counter
#
    # else: # if comment consists of more than one word
        # topics = ldamodel.get_document_topics(dictionary.doc2bow(indexed_comment[1].strip().split()),per_word_topics=True) # get per-word topic probabilities for the document
    #
        # for wt_tuple in topics[1]: # iterate over the word-topic assignments
        #
        #     if len(wt_tuple[1]) != 0: # if the model has predictions for the specific word
        #
        #         # record the most likely topic according to the trained model
        #         dxt[wt_tuple[1][0],0] += 1
        #         analyzed_comment_length += 1 # update word counter
#
#
#     if analyzed_comment_length > 0: # if the model had predictions for at least some of the words in the comment
#
#         dxt = (float(1) / float(analyzed_comment_length)) * dxt # normalize the topic contribution using comment length
#
#     else: # if the model had no reasonable topic proposal for any of the words in the comment
#
#         no_predictions = 1 # update the no_predictions counter
#
#     return (dxt,indexed_comment[2],no_predictions)
#
# ### Define the main function for multi-core calculation of topic contributions
#
# def Topic_Contribution_Multicore(path,output_path,dictionary,ldamodel,relevant_year,cumm_rel_year,num_topics):
#
#     # timer
#     print("Started calculating topic contribution at " + time.strftime('%l:%M%p'))
#
#     ## check for the existence of the preprocessed dataset
#
#     if not Path(path+'/lda_prep').is_file():
#         raise Exception('The preprocessed data could not be found')
#
#     ## initialize a dictionary for yearly topic contributions
#
#     Yearly_Running_Sums = {}
#     no_predictions_count = {}
#
#     ## Create shared counters for comments for which the model has no reasonable prediction whatsoever
#
#     for i in range(len(cumm_rel_year)):
#         Yearly_Running_Sums[i] = np.zeros([num_topics,1])
#         no_predictions_count[i] = 0
#
#     ## read and index comments
#
#     indexed_dataset = Get_Indexed_Dataset(path,cumm_rel_year)
#
#     ## define the function for spawning processes to perform the calculations in parallel
#
#     def testfunc(indexed_dataset,dictionary,ldamodel,num_topics):
#         pool = multiprocessing.Pool(processes=CpuInfo())
#         func = partial(Topic_Asgmt_Retriever_Multi, dictionary=dictionary,ldamodel=ldamodel,num_topics=num_topics)
#         results = pool.map(func=func,iterable=indexed_dataset)
#         pool.close()
#         pool.join()
#         return results
#
#     ## call the multiprocessing function on the dataset
#
#     results = testfunc(indexed_dataset,dictionary,ldamodel,num_topics)
#
#     ## Add comment-specific topic contributions to the relevant yearly sum
#
#     for comment in results: # for each comment
#         Yearly_Running_Sums[comment[1]] = Yearly_Running_Sums[comment[1]] + comment[0]
#         # update the counter for comments for which the model had no reasonable prediction
#         no_predictions_count[comment[1]] += comment[2]
#
#     # normalize contributions using the number of documents per year
#
#     yearly_output = []
#     for i in range(len(cumm_rel_year)):
#         Yearly_Running_Sums[i] = ( float(1) / (float(relevant_year[i]) - no_predictions_count[i] )) * Yearly_Running_Sums[i]
#         yearly_output.append(Yearly_Running_Sums[i])
#
#     yearly_output = np.asarray(yearly_output)
#
#     # save the yearly topic contribution matrix to disk
#
#     np.savetxt(output_path+"/yr_topic_cont", yearly_output) # save the topic contribution matrix to file
#
#     # timer
#     print("Finished calculating topic contributions at "+time.strftime('%l:%M%p'))
#
#     return yearly_output,indexed_dataset

### Function that checks for a topic contribution matrix on file and calls for its calculation if there is none

def Get_Topic_Contribution(dictionary, ldamodel, relevant_year, cumm_rel_year,
                           num_topics=num_topics, path=path,
                           output_path=output_path):

    # check to see if topic contributions have already been calculated
    if not Path(output_path+'/yr_topic_cont').is_file(): # if not

        yr_topic_cont, indexed_dataset = Topic_Contribution_Multicore(path,output_path,dictionary,ldamodel,relevant_year,cumm_rel_year,num_topics) # calculate the contributions
        np.savetxt(output_path+"/yr_topic_cont", yr_topic_cont) # save the topic contribution matrix to file

        return yr_topic_cont, indexed_dataset

    else: # if there are records on file

        # ask if the contributions should be loaded or calculated again
        Q = raw_input('Topic contribution estimations were found on file. Do you wish to delete them and calculate contributions again? [Y/N]')

        if Q == 'Y' or Q == 'y': # re-calculate

            yr_topic_cont, indexed_dataset = Topic_Contribution_Multicore(path,output_path,dictionary,ldamodel,relevant_year,cumm_rel_year,num_topics) # calculate the contributions
            np.savetxt(output_path+"/yr_topic_cont", yr_topic_cont) # save the topic contribution matrix to file

            return yr_topic_cont, indexed_dataset

        if Q == 'N' or Q == 'n': # load from file

            print("Loading topic contributions and indexed dataset from file")

            indexed_dataset = Get_Indexed_Dataset(path,cumm_rel_year)
            yr_topic_cont = np.loadtxt(output_path+"/yr_topic_cont")

            return yr_topic_cont, indexed_dataset

        else: # if the answer is neither yes, nor no
            print("Operation aborted. Please note that loaded topic contribution matrix and indexed dataset are required for determining top topics and sampling comments.")
            pass

### Define a function for plotting the temporal trends in the top topics

def Plotter(report, yr_topic_cont, name):

    plotter = []
    for topic in report:
        plotter.append(yr_topic_cont[:,topic].tolist())

    plots = {}
    for i in range(len(report.tolist())):
        plots[i]= plt.plot(range(1,len(plotter[0])+1),plotter[i],label='Topic '+str(report[i]))
    plt.legend(loc='best')
    plt.xlabel('Year (2006-'+str(2006+len(plotter[0])-1)+')')
    plt.ylabel('Topic Probability')
    plt.title('Contribution of the top topics to the LDA model for 2006-'+str(2006+len(plotter[0])-1))
    plt.grid(True)
    plt.savefig(name)
    plt.show()

### function for calculating theta ###

### A function that returns top topic probabilities for a given document (in non-zero)

def Get_LDA_Model(indexed_document,ldamodel,report):

    # info('function Topic_Asgmt_Retriever') # uncomment if you wish to observe which worker is processing a specific comment

    topics = ldamodel.get_document_topics(indexed_document[1],minimum_probability=None) # get topic probabilities for the document

    # create a tuple including the comment index, the likely top topics and the contribution of each topic to that comment if it is non-zero
    rel_probs = [(indexed_document[0],topic,prob) for topic,prob in topics if topic in report and prob > 1e-8]

    if len(rel_probs) > 0: # if the comment showed significant contribution of at least one top topic

        return rel_probs # return the the tuples (return None otherwise)

### Function for multi-core processing of comment-top topic probabilities

### IDEA: Add functionality for choosing a certain year (or interval) for which we ask the program to sample comments. Should be simple (indexed_dataset[2])

def Top_Topics_Theta_Multicore(indexed_dataset,report,dictionary,ldamodel,min_comm_length):

    # timer
    print("Started calculating theta at " + time.strftime('%l:%M%p'))

    ## filter dataset comments based on length and create a BOW for each comment

    dataset = [] # initialize dataset

    for document in indexed_dataset: # for each comment in the indexed_dataset

        if min_comm_length == None: # if not filtering based on comment length

            # add a tuple including comment index, bag of words representation and relevant year to the dataset
            dataset.append((document[0],dictionary.doc2bow(document[1].strip().split()),document[2]))

        else: # if filtering based on comment length

            if len(document[1].strip().split()) > min_comm_length: # filter out short comments

                # add a tuple including comment index, bag of words representation and relevant year to the dataset
                dataset.append((document[0],dictionary.doc2bow(document[1].strip().split()),document[2]))

    ## define the function for spawning processes to perform the calculations in parallel

    def theta_func(dataset,ldamodel,report):
        pool = multiprocessing.Pool(processes=CpuInfo())
        func = partial(Get_LDA_Model, ldamodel=ldamodel, report=report)
        theta = pool.map(func=func,iterable=dataset)
        pool.close()
        pool.join()
        return theta

    ## call the multiprocessing function on the dataset

    theta_with_none = theta_func(dataset, ldamodel, report)

    ## flatten the list and get rid of 'None's

    theta = []
    for comment in theta_with_none:
        if comment is not None:
            for item in comment:
                theta.append(item)

    return theta

### Function that calls for calculating, re-calculating or loading theta estimations for top topics

def Get_Top_Topic_Theta(indexed_dataset, report, dictionary, ldamodel,
                        min_comm_length=min_comm_length, path=path,
                        output_path=output_path):

    # check to see if theta for top topics has already been calculated

    if not Path(output_path+'/theta').is_file(): # if not

        theta = Top_Topics_Theta_Multicore(indexed_dataset,report,dictionary,ldamodel,min_comm_length) # calculate theta

        # save theta to file
        with open(output_path+'/theta','a+') as f:
            for element in theta:
                f.write(' '.join(str(number) for number in element) + '\n')

        return theta

    else: # if there are records on file

        # ask if theta should be loaded or calculated again
        Q = raw_input('Theta estimations were found on file. Do you wish to delete them and calculate probabilities again? [Y/N]')

        if Q == 'Y' or Q == 'y': # re-calculate

            os.remove(output_path+'/theta') # delete the old records

            theta = Top_Topics_Theta_Multicore(indexed_dataset,report,dictionary,ldamodel,min_comm_length) # calculate theta

            # save theta to file
            with open(output_path+'/theta','a+') as f:
                for element in theta:
                    f.write(' '.join(str(number) for number in element) + '\n')

            return theta

        if Q == 'N' or Q == 'n': # load from file

            print("Loading theta from file")

            with open(output_path+'/theta','r') as f:
                theta = [tuple(map(float, number.split())) for number in f]

            return theta

        else: # if the answer is neither yes, nor no

            print("Operation aborted. Please note that loaded theta is required for sampling top comments.")

### Defines a function for finding the [sample_comments] most representative length-filtered comments for each top topic

def Top_Comment_Indices(theta,report,sample_comments):

    top_topic_probs = {} # initialize a dictionary for all top comment indices
    sampled_indices = {} # initialize a dictionary for storing sampled comment indices
    sampled_probs = {} # initialize a list for storing top topic contribution to sampled comments

    for topic in report: # for each top topic
        # find all comments with significant contribution from that topic
        top_topic_probs[topic] = [element for element in theta if element[1] == topic]
        top_topic_probs[topic] = sorted(top_topic_probs[topic], key=lambda x: x[2],reverse=True) # sort them based on topic contribution

        # find the [sample_comments] comments for each top topic that show the greatest contribution
        sampled_indices[topic] = []
        sampled_probs[topic] = []
        for element in top_topic_probs[topic][:min(len(top_topic_probs[topic]),sample_comments)]:
            sampled_indices[topic].append(element[0]) # record the index
            sampled_probs[topic].append(element[2]) # record the contribution of the topic

    return sampled_indices,sampled_probs

### retrieve the original text of sampled comments and write them to file along with the relevant topic ID

# IDEA: Should add the possibility of sampling from specific year(s)

def Get_Top_Comments(report, cumm_rel_year, theta, path=path, output_path=output_path,
                     sample_comments=sample_comments, stop=stop):

    # timer
    print("Started sampling top comments at " + time.strftime('%l:%M%p'))

    # find the top comments associated with each top topic
    sampled_indices,sampled_probs = Top_Comment_Indices(theta,report,sample_comments)

    if not Path(path+'/original_comm').is_file(): # if the original relevant comments are not already available on disk, read them from the original compressed files

        # json parser
        decoder = json.JSONDecoder(encoding='utf-8')

        ## iterate over files in directory to find the relevant documents

        sample = 0 # counting the number of sampled comments
        counter = 0 # counting the number of all processed comments
        year_counter = 0 # the first year in the corpus (2006)

        # check for the presence of data files
        if not glob.glob(path+'/*.bz2'):
            raise Exception('No data file found')

        # open a CSV file for recording sampled comment values
        with open(output_path+'/sample_ratings.csv','a+b') as csvfile:
            writer = csv.writer(csvfile) # initialize the CSV writer
            writer.writerow(['number','index','topic','contribution','values','consequences','preferences','interpretability']) # write headers to the CSV file

        # iterate through the files in the 'path' directory in alphabetic order
        for filename in sorted(os.listdir(path)):

            # only include relevant files
            if os.path.splitext(filename)[1] == '.bz2' and 'RC_' in filename:

                ## prepare files

                # open the file as a text file, in utf8 encoding
                fin = bz2.BZ2File(filename,'r')

                # create a file to write the sampled comments to
                fout = open(output_path+'/sampled_comments','a+')

                # open CSV file to write the sampled comment data to
                csvfile = open(output_path+'/sample_ratings.csv','a+b')
                writer = csv.writer(csvfile) # initialize the CSV writer

                ## read data

                for line in fin: # for each comment

                    # parse the json, and turn it into regular text
                    comment = decoder.decode(line)
                    original_body = HTMLParser.HTMLParser().unescape(comment["body"]) # remove HTML characters

                    # filter comments by relevance to the topic
                    if len(GAYMAR.findall(original_body)) > 0 or len(MAREQU.findall(original_body)) > 0:

                        # clean the text for LDA
                        body = LDA_clean(original_body,stop)

                        # if the comment body is not empty after preprocessing
                        if body.strip() != "":

                            counter += 1 # update the counter

                            # update year counter if need be
                            if counter-1 >= cumm_rel_year[year_counter]:
                                year_counter += 1

                            for topic,indices in sampled_indices.iteritems():
                                if counter-1 in indices:

                                    # remove mid-comment lines and set encoding
                                    original_body = original_body.replace("\n","")
                                    original_body = original_body.encode("utf-8")

                                    # update the sample counter
                                    sample += 1

                                    # print the sample number to file
                                    print(sample,file=fout)

                                    # print relevant year to file
                                    print('Year: '+str(2006+year_counter),file=fout)

                                    # print the topic to file
                                    print('Topic '+str(topic),file=fout)

                                    # print the topic contribution to the comment to file
                                    itemindex = sampled_indices[topic].index(counter-1)
                                    print('Contribution: '+str(sampled_probs[topic][itemindex]),file=fout)

                                    # print the comment to file
                                    print(" ".join(original_body.strip().split()),file=fout)

                                    # print the values to CSV file
                                    writer.writerow([sample,counter-1,topic,sampled_probs[topic][itemindex]])

                                    break # if you found the index in one of the topics, no reason to keep looking

                # close the files to save the data
                fin.close()
                fout.close()
                csvfile.close()

        # timer
        print("Finished sampling top comments at " + time.strftime('%l:%M%p'))

    else: # if a file containing only the original relevant comments is available on disk

        with open(path+'/original_comm','a+') as fin, open(output_path+'/sample_ratings.csv','a+b') as csvfile, open(output_path+'/sampled_comments','a+') as fout: # determine the I/O files

            sample = 0 # initialize a counter for the sampled comments
            year_counter = 0 # initialize a counter for the comment's year
            writer = csv.writer(csvfile) # initialize the CSV writer
            writer.writerow(['number','index','topic','contribution','values','consequences','preferences','interpretability']) # write headers to the CSV file

            for comm_index,comment in enumerate(fin): # iterate over the original comments

                for topic,indices in sampled_indices.iteritems():
                    if comm_index in indices:

                        # update the year counter if need be
                        if comm_index >= cumm_rel_year[year_counter]:
                            year_counter += 1

                        # update the sample counter
                        sample += 1

                        # print the sample number to file
                        print(sample,file=fout)

                        # print the relevant year to file
                        print('Year: '+str(2006+year_counter),file=fout)

                        # print the topic to output file
                        print('Topic '+str(topic),file=fout)

                        # print the topic contribution to the comment to file
                        itemindex = sampled_indices[topic].index(comm_index)
                        print('Contribution: '+str(sampled_probs[topic][itemindex]),file=fout)

                        # print the comment to output file
                        print(" ".join(comment.strip().split()),file=fout)

                        # print the values to CSV file
                        writer.writerow([sample,comm_index,topic,sampled_probs[topic][itemindex]])

                        break # if you found the index in one of the topics, no reason to keep looking

            # timer
            print("Finished sampling top comments at " + time.strftime('%l:%M%p'))
