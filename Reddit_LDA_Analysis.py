#!/usr/bin/python27
# -*- coding: utf-8 -*-

### import the required modules and functions

from __future__ import print_function
from config import *
from Utils import *

### set default file encoding
reload(sys)
sys.setdefaultencoding('utf-8')

### Create directory for storing the output if it does not already exist
if not os.path.exists(output_path):
    print("Creating directory to store the output")
    os.makedirs(output_path)

### Write hyperparameters and performance to file

Write_Performance()

### call the parsing function

# NOTE: If NN = False, will pre-process data for LDA.
# NOTE: If write_original = True, the original text of a relevant comment - without preprocessing - will be saved to a separate file
# NOTE: If clean_raw = True, the compressed data files will be removed from disk after processing
# NOTE: Relevance filters can be changed from Utils.py. Do not forget to change the Parser function accordingly

Parse_Rel_RC_Comments(vote_counting=True,download_raw=False)

## call the function for calculating the percentage of relevant comments

# Perc_Rel_RC_Comment(path)

### create training and evaluation sets

if not ENTIRE_CORPUS:
    select_random_comments()

## Determine the comments that will comprise each set

# NOTE: If NN = False, will create sets for LDA.

Define_Sets(regression=False)

## read the data and create the vocabulary and the term-document matrix

# NOTE: Needs loaded sets. Use Define_Sets() before running this function even if prepared sets exist on file

dictionary, corpus, eval_comments, train_word_count, eval_word_count = LDA_Corpus_Processing()

### Train and Test the LDA Model ###

## create a seed for the random state generator

seed = np.random.RandomState(0)

## determine the number of CPU workers for parallel processing (set to CpuInfo() to automatically choose the optimal number)

workers = CpuInfo()

### Train or load a trained model

if not Path(path+'/RC_LDA_'+str(num_topics)+'_'+str(ENTIRE_CORPUS)+'.lda').is_file(): # if there are no trained models, train on the corpus

    # timer
    print("Started training LDA model at "+time.strftime('%l:%M%p'))

    # define and train the LDA model
    Lda = gensim.models.ldamulticore.LdaMulticore
    ldamodel = Lda(corpus, workers = workers, num_topics=num_topics, id2word = dictionary, iterations=iterations, alpha=alpha, eta=eta, random_state=seed, minimum_probability=minimum_probability, per_word_topics=True, minimum_phi_value=minimum_phi_value)
    ldamodel.save('RC_LDA_'+str(num_topics)+'_'+str(ENTIRE_CORPUS)+'.lda') # save learned model to file for future use

    # timer
    print("Finished training model at "+time.strftime('%l:%M%p'))

else: # if there is a trained model, load it from file

    print("Loading the trained LDA model from file")

    ldamodel = gensim.models.LdaMulticore.load(path+'/RC_LDA_'+str(num_topics)+'_'+str(ENTIRE_CORPUS)+'.lda')

### calculate a lower bound on per-word perplexity for training and evaluation sets

# NOTE: This function writes the estimates after calculation to the file "perf"
# NOTE: This is a slow, serial function with no method for looking for previous estimates. Check the disk manually and comment out if estimates already exist

train_per_word_perplex,eval_per_word_perplex = Get_Perplexity(ldamodel,corpus,eval_comments,training_fraction,train_word_count,eval_word_count)

### Determine Top Topics Based on Contribution to the Model ###

# NOTE: There is a strict dependency hierarchy between the functions that come in this section and the next. They should be run in the order presented

### Calculate the number of relevant comments by year

relevant_year,cumm_rel_year = Yearly_Counts()

### go through the corpus and calculate the contribution of each topic to comment content in each year

## Technical comments

# NOTE: The contribution is calculated over the entire dataset, not just the training set, but will ignore words not in the dictionary
# NOTE: Percentage of contributions is relative to the parts of corpus for which there WAS a reasonable prediction based on the model
# NOTE: For the LDA to give reasonable output, the number of topics given to this function should not be changed from what it was during model training
# NOTE: Serial, threaded and multicore (default) versions of this function are available (See Utils.py)
# NOTE: Even with multiprocessing, this function can be slow proportional to the number of top topics, as well as the size of the dataset

## Load or calculate topic distributions and create an enhanced version of the entire dataset

yr_topic_cont, indexed_dataset = Get_Topic_Contribution(dictionary, ldamodel,
    relevant_year, cumm_rel_year)

### Determine the Top Topics

avg_cont = np.empty(num_topics) # initialize a vector for average topic contribution
for i in range(num_topics):
    avg_cont[i] = np.mean(yr_topic_cont[:,i]) # calculate average topic contribution

## Find the indices of the [sample_topics] fraction of topics that have the greatest average contribution to the model

top_topic_no = int(ceil(sample_topics*num_topics))
report = avg_cont.argsort()[-top_topic_no:][::-1]

## Plot the temporal trends in the top topics

# NOTE: The resulting figure needs to be closed before functions after this point are run

Plotter(report,yr_topic_cont,output_path+'/Temporal_Trend.png')

## Find the top words associated with top topics and write them to file

with open(output_path+'/top_words','a+') as f: # create a file for storing the high-probability words
    for topic in report:
        print(topic,file=f)
        output = ldamodel.show_topic(topic,topn=topn)
        print(output,file=f)

### Find the most Representative Comments for the Top Topics ###

### Retrieve the probability assigned to top topics for comments in the dataset

# NOTE: This function only outputs the probabilities for comments of length at least [min_comm_length] with non-zero probability assigned to at least one top topic

theta = Get_Top_Topic_Theta(indexed_dataset, report, dictionary, ldamodel)

### for the top topics, choose the [sample_comments] comments that reflect the greatest contribution of those topics and write them to file

# NOTE: If write_original was set to False during the initial parsing, this function will require the original compressed data files (and will be much slower). If not in the same directory as this file, change the "path" argument

Get_Top_Comments(report, cumm_rel_year, theta)
