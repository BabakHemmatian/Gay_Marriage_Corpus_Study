#!/python27
# -*- coding: utf-8 -*-
# import the required modules and functions
from __future__ import print_function
from __future__ import division
from json import JSONDecoder    # imports the code to parse json
import os
import sys
import codecs
import numpy as np
from scipy import sparse
import lda
import nltk
import csv
import time
from scipy.sparse import csr_matrix, vstack, hstack
# nltk.download('wordnet')
# nltk.download('stopwords')
# path to the data (same folder as this code)
file_path = os.path.abspath(sys.argv[0])
path = os.path.dirname(file_path)
# set database (RC=Reddit,Fb=Facebook)
database = "RC"
# define the set of stopwords
stop = set(nltk.corpus.stopwords.words('english'))
keepers = ["how","should","can","will","just","so","few","more","most","why","how","all","any","about","against","because","as","have","has","had","is","are","was","were","be","been","being","they","them","their","theirs","themselves","i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves"]
stop = [word.encode('ascii') for word in stop if word not in keepers]
# if a cooccurrence matrix is already available, uncomment the next line to read that instead of creating a new matrix
# my_matrix = numpy.loadtxt(open("matrix.txt","rb"),delimiter=",",skiprows=0)
# initialize the cooccurrence matrix
cooc_mat = np.zeros((1,1))
cooc_mat = csr_matrix(cooc_mat) #csr format allows regular numpy notation for assigning values
# initialize the list of words
bag_of_words = []
# if a vocabulary is already available, uncomment the next line to read that instead of creating a new dictionary
# for word in csv.reader(open("vocab.csv")):
#     bag_of_words.append(word)
# This is the json parser
decoder = JSONDecoder(encoding='utf-8')
counter = 0
# find preprocessed data
for filename in os.listdir(path):
    # for Reddit database
    if database == "RC":
        if "prep" in filename and "RC" in filename:
            print(filename+time.strftime('%l:%M%p'))
            fin = codecs.open(filename,mode='r',encoding='utf8')
            # fix the formatting of the data lines
            for line in fin:
                if line != "":
                    line = line.encode('utf8','ignore')
                    line = line.replace("\n","")
                    line = line.replace("\r","")
                    counter += 1
                    # form a document-term cooccurrence matrix
                    if counter > 1:
                        new_doc = csr_matrix(np.zeros((1,cooc_mat.shape[1])))
                        cooc_mat = sparse.vstack([cooc_mat,new_doc])
                    temp = line.split(" ")
                    for word in temp:
                        if "http" not in word and word.isdigit() == False and word not in stop and word.isalnum():
                            if bag_of_words == []:
                                bag_of_words.append(word)
                                cooc_mat[0][0]=1
                            else:
                                if word not in bag_of_words:
                                    bag_of_words.append(word)
                                    new_word = csr_matrix(np.zeros((cooc_mat.shape[0],1)))
                                    cooc_mat = sparse.hstack([cooc_mat,new_word])
                                    cooc_mat = cooc_mat.tocsr()
                                    freq = cooc_mat[cooc_mat.shape[0]-1,cooc_mat.shape[1]-1]
                                    cooc_mat[cooc_mat.shape[0]-1,cooc_mat.shape[1]-1] = freq + 1
                                else:
                                    freq = cooc_mat[cooc_mat.shape[0]-1,bag_of_words.index(word)]
                                    cooc_mat[cooc_mat.shape[0]-1,bag_of_words.index(word)] = freq + 1
    # if database is from Facebook
    elif database == "Fb":
        if "prep" in filename and "RC" not in filename:
            fin = codecs.open(filename,mode='r',encoding='utf8')
            # fix the formatting of the data lines
            for line in fin:
                if line != "":
                    line = line.encode('utf8','ignore')
                    line = line.replace("\n","")
                    line = line.replace("\r","")
                    counter += 1
                    # form a document-term cooccurrence matrix
                    if counter > 1:
                        new_doc = csr_matrix(np.zeros((1,cooc_mat.shape[1])))
                        cooc_mat = sparse.vstack([cooc_mat,new_doc])
                    temp = line.split(" ")
                    for word in temp:
                        if "http" not in word and word.isdigit() == False and word not in stop and word.isalnum():
                            if bag_of_words == []:
                                bag_of_words.append(word)
                                cooc_mat[0][0]=1
                            else:
                                if word not in bag_of_words:
                                    bag_of_words.append(word)
                                    new_word = csr_matrix(np.zeros((cooc_mat.shape[0],1)))
                                    cooc_mat = sparse.hstack([cooc_mat,new_word])
                                    cooc_mat = cooc_mat.tocsr()
                                    freq = cooc_mat[cooc_mat.shape[0]-1,cooc_mat.shape[1]-1]
                                    cooc_mat[cooc_mat.shape[0]-1,cooc_mat.shape[1]-1] = freq + 1
                                else:
                                    freq = cooc_mat[cooc_mat.shape[0]-1,bag_of_words.index(word)]
                                    cooc_mat[cooc_mat.shape[0]-1,bag_of_words.index(word)] = freq + 1
cooc_mat = cooc_mat.astype(np.int32)
print("Finished creating the cooccurrence matrix at "+time.strftime('%l:%M%p'))
# print the cooccurrence matrix to file for future use
cooc_mat = np.asarray(cooc_mat.todense())
np.savetxt('cooc_mat.txt', cooc_mat, delimiter = ',')
print("Finished writing cooccurrence matrix to file at "+time.strftime('%l:%M%p'))
# print the vocabulary to file for future use
w = csv.writer(open("vocab.csv", "w"))
for key, val in bag_of_words.items():
    w.writerow([key, val])
print("Finished writing vocabulary to file at "+time.strftime('%l:%M%p'))
# define the number of topics
n_topics = 100
# run LDA
model = lda.LDA(n_topics=n_topics,n_iter=1500, random_state=1)
model.fit(cooc_mat)
topic_word = model.topic_word_
t_w_assignment = model.nzw_
perc_assigned = np.array(np.zeros((n_topics,1)),dtype=float)
n_top_words = 20
# determine the percentage of all words in the corpus that is assigned to a certain topic
for i in range(0,len(t_w_assignment)):
    perc_assigned[i] = sum(t_w_assignment[i])/sum(sum(t_w_assignment))
# print out the most likely words from each topic alongside the percentage of words assigned to it
for i, topic_dist in enumerate(topic_word):
    if perc_assigned[i]>= 0.01:
        topic_words = np.array(bag_of_words)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i+1, ' '.join(topic_words)))
        print("topic"+str(i+1)+":"+str(perc_assigned[i]))
