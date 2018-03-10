import nltk
import os
import re
import sys

# NOTE: This file only contains the hyperparameters at the most abstract level,
# those that are most likely to be tuned by the user. See relevant functions
# in Utils.py for finer tuning of parameters.
# NOTE: If you wish to keep the following values as defaults, but try out other
# values, you can override the defaults by assinging variables in lda_config.py

### determine hyperparameters ###

### Model choice hyperparameters

NN = False # For development purposes. Should always be set to False for LDA
ENTIRE_CORPUS = True # Are we using a random subset of comments, or the whole
# dataset? The names of output files will include the value of this variable
OVERWRITE = False # Overwrites existing sampled comment indices. Only matters
# if ENTIRE_CORPUS = False
DOWNLOAD_RAW = True # If a raw data file is not available on disk, download it
# if DOWNLOAD_RAW set to True
CLEAN_RAW = True # After parsing, delete the raw data file from disk
vote_counting = False # Count number of upvotes when parsing
WRITE_ORIGINAL = True # Write original comments to file when parsing

### Pre-processing hyperparameters
MaxVocab = 2000000 # maximum size of the vocabulary
FrequencyFilter = 1 # tokens with a frequency equal or less than this number will be filtered out of the corpus
no_below = 5 # tokens that appear in less than this number of documents in
# corpus will be filtered out
no_above = 0.99 # tokens that appear in more than this fraction of documents in
# corpus will be filtered out
training_fraction = 0.99 # what percentage of data will be used for training.
# The rest of the dataset will be used as an evaluation set for calculating
# perplexity
calculate_perc_rel = False # whether the percentage of relevant comments from
# each year should be calculated and written to file

### LDA hyperparameters
n_random_comments = 1500 # number of comments to sample from each year for
# training. Only matters if ENTIRE_CORPUS = False.
iterations = 1000 # number of times LDA posterior distributions will be sampled
num_threads = 5 # number of threads used for parallelized processing of comments
# Only matters if using _Threaded functions
num_topics = 75 # number of topics to be generated in each LDA sampling
alpha = 0.1 # determines how many high probability topics will be assigned to a
# document in general (not to be confused with NN l2regularization constant)
# @BabakHemmatian, will this change be problematic
#minimum_probability = 0.01 # minimum acceptable probability for an output topic
# across corpus
minimum_probability = 1e-8
eta = 0.1 # determines how many high probability words will be assigned to a
# topic in general
minimum_phi_value = 0.01 # determines the lower bound on per-term topic
# probability. Only matters if per_word_topics = True.
calculate_perplexity = True # whether perplexity should be calculated for the
# LDA model

### Sampling hyperparameters
sample_topics = 0.1 # percentage of topics that will be selected for reporting
# based on average yearly contribution
topn = 80 # the number of high-probability words for each topic to be exported
# NOTE: Many of the words will inevitably be high probability general
# non-content and non-framing words. So topn should be set to significantly
# higher than the number of relevant words you wish to see
sample_comments = 100 # number of comments that will be sampled from top topics
min_comm_length = 40 # the minimum acceptable number of words in a sampled
# comment. Set to None for no length filtering
# Determines how topic contributions are calculated. When set to True, the
# topic of each word is set to be simply the most probable word. When False,
# the topic of each word is set to the entire probability distribution over
# num_topics topics.
one_hot_topic_contributions=True

### Paths

## where the data is
# NOTE: if not fully available on file, set Download for Parser function to
# True (source: http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable
# accordingly
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)

## Year/month combinations to get Reddit data for
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2006,2018) # year range
for year in years:
    for month in months:
        if year==2017 and month==10: # till Sep 2017
            break
        dates.append((year,month))

## where the output will be stored
# NOTE: To avoid confusion between different kinds of models, record the
# variables most important to your iteration in the folder name
# Force this import so output_path is correctly set
from lda_config import ENTIRE_CORPUS
output_path = path + "/LDA_"+str(ENTIRE_CORPUS)+"_"+str(num_topics)

### Preprocessing ###

### determine the set of stopwords used in preprocessing

keepers = ["how","should","should've","could","can","need","needn","why","few","more","most","all","any","against","because","ought","must","mustn","mustn't","shouldn","shouldn't","couldn't","couldn","shan't", "needn't"]
stop = []
for word in set(nltk.corpus.stopwords.words('english')):
    if word not in keepers:
        stop.append(str(word))

regex=re.compile("(^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$)|(^(?=.*marriage equality|.*equal marriage).*$)", re.I)
