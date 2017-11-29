#!/usr/bin/python27
# -*- coding: utf-8 -*-
### import the required modules and functions
from __future__ import print_function
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
from numpy import sign
# uncomment the next few lines if nltk packages are not locally installed
# nltk.download('stopwords')
# nltk.download('punkt')
### set hyperparameters
# set default file encoding
reload(sys)
sys.setdefaultencoding('utf8')
### Preprocessing
## determine the set of stopwords used in preprocessing
stop = set(nltk.corpus.stopwords.words('english'))
exclude = set(string.punctuation)
## define the preprocessing function to add padding and remove punctuation, special characters and stopwords
def clean(text):
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
        if punc_free != "" and punc_free!= " ":
            padded = punc_free+" *STOP*"
            if index+1 == len(text):
                padded = padded+" *STOP2*"
            cleaned.append(padded)
        elif (punc_free == "" or punc_free == " ") and len(text)!=1 and len(cleaned)!=0 and index+1 == len(text):
            cleaned[-1] = cleaned[-1]+" *STOP2*"
    return cleaned
### unpack data
# json parser
decoder = json.JSONDecoder(encoding='utf-8')
# where the data is
file_path = os.path.abspath(sys.argv[0])
path = os.path.dirname(file_path)
# define the relevance filter (added marriage equality to the list of keywords 11.20.17)
def getFilterBasicRegex():
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)
def getFilterEquRegex():
    return re.compile("^(?=.*marriage equality|.*equal marriage).*$", re.I)
GAYMAR = getFilterBasicRegex()
MAREQU = getFilterEquRegex()
# import the pre-trained PUNKT tokenizer for determining sentence boundaries
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
# initialize containers for number of comments and indices related to each month
timedict = dict()
### parse data
# timer
print("Started parsing at " + time.strftime('%l:%M%p'))
# iterate over files in directory to preprocess the text and record the votes
counter = 0 # counting the number of all processed comments
for filename in sorted(os.listdir(path)):
    # only include relevant files
    if os.path.splitext(filename)[1] == '.bz2' and 'RC_' in filename:
        # open the file as a text file, in utf8 encoding
        fin = bz2.BZ2File(filename,'r')
        # create a file to write the processed text to
        fout = open("nn_prep.txt",'a+')
        # create a file to store the relevant indices for each month
        ccount = open("RC_Count_List.txt",'a+')
        # every line is a comment
        for line in fin:
            # parse the json, and turn it to regular text
            comment = decoder.decode(line)
            body = HTMLParser.HTMLParser().unescape(comment["body"])
            # filter comments by relevance to the topic
            if len(GAYMAR.findall(body)) > 0 or len(MAREQU.findall(body)) > 0:
                counter += 1 # update the counter
                # record the number of documents by year and month
                created_at = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime('%Y-%m')
                if created_at not in timedict:
                    timedict[created_at] = 1
                else:
                    timedict[created_at] += 1
                ## preprocess the comments
                # tokenize the sentences
                body = sent_detector.tokenize(body)
                # clean the the text, remove mid-comment lines and set encoding
                body = clean(body)
                # print the processed comment to file
                if body != []:
                    for sen in body:
                        sen = sen.replace("\n","")
                        sen = sen.encode("utf-8")
                        print(" ".join(sen.split()), end=" ", file=fout)
                # ensure that each comment is on a separate line
                print("\n",end="",file=fout)
        # write the comment indices for each month to file
        print(counter,file=ccount)
        # close the files to save the data
        fin.close()
        fout.close()
        ccount.close()
        # timer
        print("Finished parsing "+filename+" at " + time.strftime('%l:%M%p'))
# timer
print("Finished parsing at " + time.strftime('%l:%M%p'))
### write the distribution of comments by month to file
fcount = open("RC_Count_Dict.txt",'a+')
for month,docs in timedict.iteritems():
    print(month+" "+str(docs),end='\n',file=fcount)
fcount.close
### determine what percentage of the posts in each year was relevant
# load the toal monthly counts into a dictionary
d = {}
with open("RC_Count_Total.txt") as f:
    for line in f:
        line = line.replace("\n","")
        if line != "":
            (key, val) = line.split("  ")
            d[key] = int(val)
    f.close
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
# calculate the percentage of comments in each year that was relevant
perc_rel = {}
rel = open("perc_rel.txt",'a+')
for key in relevant_year:
    perc_rel[key] = float(relevant_year[key]) / float(total_year[key])
print(sorted(perc_rel.items()),file=rel)
rel.close
