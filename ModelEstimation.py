from __future__ import print_function
from collections import defaultdict, OrderedDict
import csv
from functools import partial
import operator
import numpy as np
import gensim
import pickle
from math import ceil, floor
import matplotlib.pyplot as plt
import multiprocessing
import os
from pathlib2 import Path
from random import sample
import time
from config import *
from reddit_parser import Parser
parser_fns=Parser().get_parser_fns()
from Utils import *

## wrapper function for calculating topic contributions to a comment
def Topic_Asgmt_Retriever_Multi_wrapper(args):
    indexed_comment, kwargs=args
    data_for_R = LDAModel(**kwargs).Topic_Asgmt_Retriever_Multi(indexed_comment)
    return data_for_R

## define the function for spawning processes to perform the calculations in parallel
def theta_func(dataset, ldamodel, report):
    pool = multiprocessing.Pool(processes=CpuInfo())
    func = partial(Get_LDA_Model, ldamodel=ldamodel, report=report)
    theta = pool.map(func=func,iterable=dataset)
    pool.close()
    pool.join()
    return theta

# A function that returns top topic probabilities for a given document
# (in non-zero)
def Get_LDA_Model(indexed_document, ldamodel, report):
    # get topic probabilities for the document
    topics = ldamodel.get_document_topics(indexed_document[1],
        minimum_probability=minimum_probability)

    # create a tuple including the comment index, the likely top topics and the
    # contribution of each topic to that comment if it is non-zero
    rel_probs = [(indexed_document[0],topic,prob) for topic,prob in topics if
                 topic in report and prob > 1e-8]

    if len(rel_probs) > 0: # if the comment showed significant contribution of
    # at least one top topic
        return rel_probs # return the the tuples (return None otherwise)

# Define a class of vectors in basic C that will be shared between multi-core
# prcoesses for calculating topic contribution
class Shared_Contribution_Array(object):
    ## Shared_Contribution_Array attributes
    def __init__(self, num_topics=num_topics):
        self.val = multiprocessing.RawArray('f', np.zeros([num_topics,1])) # shape and data type
        self.lock = multiprocessing.Lock() # prevents different processes from writing the shared variables at the same time and mixing data up

    ## Shared_Contribution_Array update method
    def Update_Val(self, dxt):
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

class ModelEstimator(object):
    def __init__(self, all_=ENTIRE_CORPUS, MaxVocab=MaxVocab,
                 output_path=output_path, path=path, regression=False,
                 training_fraction=training_fraction, V=OrderedDict({})):
            ## ensure the arguments have the correct types and values
            assert type(path) is str
            assert 0 < training_fraction and 1 > training_fraction
            assert type(NN) is bool
            # check the given path
            if not os.path.exists(path):
                raise Exception('Invalid path')
            self.all_=all_
            self.MaxVocab=MaxVocab
            self.output_path=output_path
            self.path=path
            self.regression=regression
            self.training_fraction=training_fraction
            self.V=V # vocabulary

    ### function to determine comment indices for new training, development and test sets
    def Create_New_Sets(self, indices):
        print("Creating sets")

        # determine number of comments in the dataset
        if self.all_:
            if not self.regression: # if not doing regression on sampled comments
                num_comm = indices[-1] # retrieve the total number of comments
                indices = range(num_comm) # define sets over all comments

            else: # if doing regression on sampled comments
                # check to see if human comment ratings can be found on disk
                if not Path(self.fns["sample_ratings"]).is_file():
                    raise Exception("Human comment ratings for regressor training could not be found on file.")

                # retrieve the number of comments for which there are complete human ratings
                with open(self.fns["sample_ratings"],'r+b') as csvfile:
                    reader = csv.reader(csvfile)
                    human_ratings = [] # initialize counter for the number of valid human ratings
                    # read human data for sampled comments one by one
                    for idx,row in enumerate(reader):
                        row = row[0].split(",")
                        # ignore headers and record the index of comments that are interpretable and that have ratings for all three goal variables
                        if ( idx != 0 and (row[5] != 'N' or row[5] != 'n') and
                             row[2].isdigit() and row[3].isdigit() and
                             row[4].isdigit() ):
                            human_ratings.append(int(row[0]))

                num_comm = len(human_ratings) # the number of valid samples for network training
                indices = human_ratings # define sets over sampled comments with human ratings

        else: # if using LDA on a random subsample of the comments
            num_comm = len(indices) # total number of sampled comments

        num_train = int(ceil(training_fraction * num_comm)) # size of training set

        if isinstance(self, NNModel): # for NN
            num_remaining = num_comm - num_train # the number of comments in development set or test set
            num_dev = int(floor(num_remaining/2)) # size of the development set
            num_test = num_remaining - num_dev # size of the test set

            self.sets['dev'] = sample(indices, num_dev) # choose development comments at random
            remaining = set(indices).difference(self.sets['dev'])
            self.sets['test']  = sample(remaining,num_test) # choose test comments at random
            # use the rest as training set
            self.sets['train'] = set(remaining).difference(self.sets['test'])

            # sort the indices based on position in nn_prep
            for set_key in self.set_key_list:
                self.sets[set_key] = sorted(list(self.sets[set_key]))

            # Check dev and test sets came out with right proportions
            assert (len(self.sets['dev']) - len(self.sets['test'])) <= 1
            assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets['train']) == len(indices)

            # write the sets to file
            for set_key in self.set_key_list:
                with open(self.path+'/'+set_key+'_set_'+str(self.regression),'a+') as f:
                    for index in self.sets[set_key]:
                        print(index,end='\n',file=f)

        else: # for LDA
            num_eval = num_comm - num_train # size of evaluation set

            self.LDA_sets['eval'] = sample(indices,num_eval) # choose evaluation comments at random
            self.LDA_sets['train'] = set(indices).difference(set(self.LDA_sets['eval'])) # assign the rest of the comments to training

            # sort the indices based on position in lda_prep
            for set_key in self.LDA_set_keys:
                self.LDA_sets[set_key] = sorted(list(self.LDA_sets[set_key]))

            # Check that sets came out with right proportions
            assert len(self.LDA_sets['train']) + len(self.LDA_sets['eval']) == len(indices)

            # write the sets to file
            for set_key in self.LDA_set_keys:
                with open(self.fns["{}_set".format(set_key)],'a+') as f:
                    for index in self.LDA_sets[set_key]:
                        print(index,end='\n',file=f)

    ### function for loading, calculating, or recalculating sets
    def Define_Sets(self):
        # load the number of comments or raise Exception if they can't be found
        findices=self.fns["counts"] if self.all_ else self.fns["random_indices"]
        try:
            assert os.path.exists(findices)
        except AssertionError:
            raise Exception("File {} not found.".format(findices))

        indices=open(findices, 'r').read().split()
        indices=filter(lambda x:x.strip(), indices)
        indices=map(int, indices)

        # if indexed comments are available (NN)
        if (isinstance(self, NNModel) and
            Path(self.fns["train_set"]).is_file() and
            Path(self.fns["dev_set"]).is_file() and
            Path(self.fns["test_set"]).is_file()):

            # determine if the comments and their relevant indices should be deleted and re-initialized or the sets should just be loaded
            Q = raw_input("Indexed comments are already available. Do you wish to delete sets and create new ones [Y/N]?")

            # If recreating the sets is requested, delete the current ones and reinitialize
            if Q == "Y" or Q == "y":
                print("Deleting any existing sets and indexed comments")

                # delete previous record
                for set_key in self.set_key_list:
                    if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["indexed_{}_set".format(set_key)])
                    if Path(self.fns["{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["{}_set".format(set_key)])

                self.Create_New_Sets(indices) # create sets

            # If recreating is not requested, attempt to load the sets
            elif Q == "N" or Q == "n":
                # if the sets are found, load them
                if ( Path(self.fns["train_set"]).is_file()
                     and Path(self.fns["dev_set"]).is_file()
                     and Path(self.fns["test_set"]).is_file()
                   ):

                    print("Loading sets from file")

                    for set_key in self.set_key_list:
                        with open(self.fns["{}_set".format(set_key)],'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.sets[set_key].append(int(line))
                        self.sets[set_key] = np.asarray(self.sets[set_key])

                    # ensure set sizes are correct
                    assert len(self.sets['dev']) - len(self.sets['test']) < 1
                    assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets['train']) == len(indices)

                else: # if the sets cannot be found, delete any current sets and create new sets
                    print("Failed to load previous sets. Reinitializing")

                    # delete partial record
                    for set_key in self.set_key_list:
                        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["indexed_{}_set".format(set_key)])
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    self.Create_New_Sets(indices) # create sets

            else: # if response was something other tha Y or N
                print("Operation aborted")
                pass

        else: # no indexed comments available or not creating sets for NN
            # delete any possible partial indexed set
            if isinstance(self, NNModel):
                for set_key in self.set_key_list:
                    if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["indexed_{}_set".format(set_key)])

            # check to see if there are sets available, if so load them
            if (isinstance(self, NNModel) and
                Path(self.fns["train_set"]).is_file() and
                Path(self.fns["dev_set"]).is_file() and
                Path(self.fns["test_set"]).is_file()
               ) or (not isinstance(self, NNModel) and
                Path(self.fns["train_set"]).is_file() and
                Path(self.fns["eval_set"]).is_file()):

                print("Loading sets from file")

                if isinstance(self, NNModel): # for NN
                    for set_key in self.set_key_list:
                        with open(self.fns["{}_set".format(set_key)],'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.sets[set_key].append(int(line))
                        self.sets[set_key] = np.asarray(self.sets[set_key])

                    # ensure set sizes are correct
                    assert len(self.sets['dev']) - len(self.sets['test']) < 1
                    l=indices[-1] if (self.all_ and not self.regression) else len(indices)
                    assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets['train']) == l

                else: # for LDA
                    for set_key in self.LDA_set_keys:
                        with open(self.fns["{}_set".format(set_key)],'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.LDA_sets[set_key].append(int(line))
                        self.LDA_sets[set_key] = np.asarray(self.LDA_sets[set_key])

            else: # if not all sets are found
                if isinstance(self, NNModel): # for NN
                    # delete any partial set
                    for set_key in self.set_key_list:
                        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["indexed_{}_set".format(set_key)])
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    # create new sets
                    self.Create_New_Sets(indices)

                else: # for LDA
                    # delete any partial set
                    for set_key in self.LDA_set_keys:
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    # create new sets
                    self.Create_New_Sets(indices)

class LDAModel(ModelEstimator):
    def __init__(self, alpha=alpha, corpus=None, dictionary=None, eta=eta,
                 eval_comments=None, eval_word_count=None, fns=dict(),
                 indexed_dataset=None, iterations=iterations,
                 LDA_set_keys=[ 'train', 'eval' ], LDA_sets=None, ldamodel=None,
                 min_comm_length=min_comm_length,
                 minimum_phi_value=minimum_phi_value,
                 minimum_probability=minimum_probability, no_above=no_above,
                 no_below=no_below, num_topics=num_topics,
                 one_hot=one_hot_topic_contributions,
                 sample_comments=sample_comments, sample_topics=sample_topics,
                 top_topic_thresh=top_topic_thresh, top_topic_set=top_topic_set, stop=stop,
                 topic_cont_freq=topic_cont_freq, train_word_count=None,
                 **kwargs):
        ModelEstimator.__init__(self, **kwargs)
        self.alpha=alpha
        self.corpus=corpus
        self.dictionary=dictionary
        self.eta=eta
        self.eval_comments=eval_comments
        self.eval_word_count=eval_word_count
        self.indexed_dataset=indexed_dataset
        self.iterations=iterations
        self.LDA_set_keys=LDA_set_keys
        self.LDA_sets=LDA_sets
        if isinstance(self.LDA_sets, type(None)):
            self.LDA_sets = {key: [] for key in self.LDA_set_keys}
        self.ldamodel=ldamodel
        self.min_comm_length=min_comm_length
        self.minimum_phi_value=minimum_phi_value
        self.minimum_probability=minimum_probability
        self.no_above=no_above
        self.no_below=no_below
        self.num_topics=num_topics
        self.one_hot=one_hot
        self.sample_comments=sample_comments
        self.sample_topics=sample_topics
        self.top_topic_thresh=top_topic_thresh
        self.stop=stop
        self.topic_cont_freq=topic_cont_freq
        self.train_word_count=train_word_count
        self.fns=self.get_fns(**fns)

    def get_fns(self, **kwargs):
        fns={ "original_comm":"{}/original_comm".format(self.path),
              "lda_prep":"{}/lda_prep".format(self.path),
              "counts": parser_fns["counts"] if self.all_ else parser_fns["counts_random"],
              "random_indices": parser_fns["indices_random"],
              "train_set":"{}/LDA_train_set_{}".format(self.path, self.all_),
              "eval_set":"{}/LDA_eval_set_{}".format(self.path, self.all_),
              "corpus":"{}/RC_LDA_Corpus_{}.mm".format(self.path, self.all_),
              "eval":"{}/RC_LDA_Eval_{}.mm".format(self.path, self.all_),
              "dictionary":"{}/RC_LDA_Dict_{}.dict".format(self.path, self.all_),
              "train_word_count":"{}/train_word_count_{}".format(self.path, self.all_),
              "eval_word_count":"{}/eval_word_count_{}".format(self.path, self.all_),
              "model":"{}/RC_LDA_{}_{}.lda".format(self.path, self.num_topics, self.all_),
              "performance": "{}/Performance".format(self.output_path),
              "topic_cont":"{}/yr_topic_cont_{}-{}-{}".format(self.output_path,
                           "one-hot" if self.one_hot else "distributions",
                           "all" if self.all_ else "subsample",
                           self.topic_cont_freq),
              "theta":"{}/theta_{}-{}-{}".format(self.output_path,
                      "one-hot" if self.one_hot else "distributions",
                      "all" if self.all_ else "subsample", self.topic_cont_freq),
              "sample_keys": "{}/sample_keys.csv".format(self.output_path),
              "sample_ratings": "{}/sample_ratings.csv".format(self.output_path),
              "sampled_comments":"{}/sampled_comments".format(self.output_path),
              "popular_comments":"{}/popular_comments.csv".format(self.output_path),
              "original_comm":"{}/original_comm".format(self.path),
              "counts":"{}/RC_Count_List".format(self.path),
              "votes":"{}/votes".format(self.path),
              "data_for_R":"{}/data_for_R.csv".format(self.output_path)
            }
        for k, v in kwargs.items():
            fns[k]=v
        return fns

    ### Function for reading and indexing a pre-processed corpus for LDA
    def LDA_Corpus_Processing(self):
        # check the existence of pre-processed data and sets
        if not Path(self.fns["lda_prep"]).is_file():
            raise Exception('Pre-processed data could not be found')
        if ( not Path(self.fns["train_set"]).is_file() or
             not Path(self.fns["eval_set"]).is_file() ):
            raise Exception('Comment sets could not be found')

        # open the file storing pre-processed comments
        f = open(self.fns["lda_prep"],'r')

        # check to see if the corpus has previously been processed
        required_files = [self.fns["corpus"], self.fns["eval"],
                          self.fns["dictionary"], self.fns["train_word_count"],
                          self.fns["eval_word_count"]]
        missing_file = 0
        for saved_file in required_files:
            if not Path(saved_file).is_file():
                missing_file += 1

        # if there is a complete extant record, load it
        if missing_file == 0:
            corpus = gensim.corpora.MmCorpus(self.fns["corpus"])
            eval_comments = gensim.corpora.MmCorpus(self.fns["eval"])
            dictionary = gensim.corpora.Dictionary.load(self.fns["dictionary"])
            with open(self.fns["train_word_count"]) as g:
                for line in g:
                    if line.strip() != "":
                        train_word_count = int(line)
            with open(self.fns["eval_word_count"]) as h:
                for line in h:
                    if line.strip() != "":
                        eval_word_count = int(line)

            print("Finished loading the dictionary and the indexed corpora from file")

        # delete any incomplete corpus record
        elif missing_file > 0 and missing_file != len(required_files):
            for saved_file in required_files:
                if Path(saved_file).is_file():
                    os.remove(saved_file)
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
                if index in self.LDA_sets['train']: # if it belongs in the training set
                    document = [] # initialize a bag of words
                    if len(comment.strip().split()) == 1:
                        document.append(comment.strip())
                    else:
                        for word in comment.strip().split(): # for each word
                            document.append(word)

                    train_word_count += len(document)
                    texts.append(document) # add the BOW to the corpus

                elif index in self.LDA_sets['eval']: # if in evaluation set
                    document = [] # initialize a bag of words
                    if len(comment.strip().split()) == 1:
                        document.append(comment.strip())
                    else:
                        for word in comment.strip().split(): # for each word
                            document.append(word)

                    eval_word_count += len(document)
                    eval_comments.append(document) # add the BOW to the corpus

                else: # if the index is in neither set and we're processing the entire corpus, raise an Exception
                    if self.all_:
                        raise Exception('Error in processing comment indices')
                    continue

            # write the number of words in the frequency-filtered corpus to file
            with open(self.fns["train_word_count"],'w') as u:
                print(train_word_count,file=u)

            # write the number of words in the frequency-filtered evaluation set to file
            with open(self.fns["eval_word_count"],'w') as w:
                print(eval_word_count,file=w)

            ## create the dictionary

            dictionary = gensim.corpora.Dictionary(texts,prune_at=self.MaxVocab) # training set
            dictionary.add_documents(eval_comments,prune_at=self.MaxVocab) # add evaluation set
            dictionary.filter_extremes(no_below=self.no_below,
                                       no_above=self.no_above, keep_n=MaxVocab) # filter extremes
            dictionary.save(self.fns["dictionary"]) # save dictionary to file for future use

            ## create the Bag of Words (BOW) datasets
            corpus = [dictionary.doc2bow(text) for text in texts] # turn training comments into BOWs
            eval_comments = [dictionary.doc2bow(text) for text in eval_comments] # turn evaluation comments into BOWs
            gensim.corpora.MmCorpus.serialize(self.fns["corpus"], corpus) # save indexed data to file for future use (overwrites any previous versions)
            gensim.corpora.MmCorpus.serialize(self.fns["eval"], eval_comments) # save the evaluation set to file

            # timer
            print("Finished creating the dictionary and the term-document matrices at "+time.strftime('%l:%M%p'))

        self.dictionary=dictionary
        self.corpus=corpus
        self.eval_comments=eval_comments
        self.train_word_count=train_word_count
        self.eval_word_count=eval_word_count

    ### Train or load a trained model
    def get_model(self):
        if not Path(self.fns["model"]).is_file(): # if there are no trained models, train on the corpus
            # timer
            print("Started training LDA model at "+time.strftime('%l:%M%p'))

            ## create a seed for the random state generator
            seed = np.random.RandomState(0)

            ## determine the number of CPU workers for parallel processing (set to CpuInfo() to automatically choose the optimal number)
            workers = CpuInfo()

            # define and train the LDA model
            Lda = gensim.models.ldamulticore.LdaMulticore
            self.ldamodel = Lda(self.corpus, workers = workers,
                                num_topics=self.num_topics,
                                id2word = self.dictionary,
                                iterations=self.iterations, alpha=self.alpha,
                                eta=self.eta, random_state=seed,
                                minimum_probability=self.minimum_probability,
                                per_word_topics=True,
                                minimum_phi_value=self.minimum_phi_value)
            self.ldamodel.save(self.fns["model"]) # save learned model to file for future use

            # timer
            print("Finished training model at "+time.strftime('%l:%M%p'))

        else: # if there is a trained model, load it from file
            print("Loading the trained LDA model from file")

            self.ldamodel = gensim.models.LdaMulticore.load(self.fns["model"])

    ### Get lower bounds on per-word perplexity for training and development sets (LDA)
    def Get_Perplexity(self):
        # timer
        print("Started calculating perplexity at "+time.strftime('%l:%M%p'))

        ## calculate model perplexity for training and evaluation sets
        train_perplexity = self.ldamodel.bound(self.corpus,
            subsample_ratio = self.training_fraction)
        eval_perplexity = self.ldamodel.bound(self.eval_comments,
            subsample_ratio = 1-self.training_fraction)

        ## calculate per-word perplexity for training and evaluation sets
        train_per_word_perplex = np.exp2(-train_perplexity / self.train_word_count)
        eval_per_word_perplex = np.exp2(-eval_perplexity / self.eval_word_count)

        # timer
        print("Finished calculating perplexity at "+time.strftime('%l:%M%p'))

        ## Print and save the per-word perplexity values to file
        with open(self.fns["performance"],'a+') as perf:
            print("*** Perplexity ***",file=perf)
            print("Lower bound on per-word perplexity (using "+str(self.training_fraction)+" percent of documents as training set): "+str(train_per_word_perplex))
            print("Lower bound on per-word perplexity (using "+str(self.training_fraction)+" percent of documents as training set): "+str(train_per_word_perplex),file=perf)
            print("Lower bound on per-word perplexity (using "+str(1-self.training_fraction)+" percent of held-out documents as evaluation set): "+str(eval_per_word_perplex))
            print("Lower bound on per-word perplexity (using "+str(1-self.training_fraction)+" percent of held-out documents as evaluation set): "+str(eval_per_word_perplex),file=perf)

        return train_per_word_perplex,eval_per_word_perplex

    ### Get umass coherence values for the LDA model based on training and development sets
    def Get_Coherence(self):
        # timer
        print("Started calculating coherence at "+time.strftime('%l:%M%p'))

        ## calculate model coherence for training set
        umass = gensim.models.coherencemodel.CoherenceModel
        train_coherence = umass(model=self.ldamodel, corpus=self.corpus, coherence='u_mass')
        umass_train_value = train_coherence.get_coherence()

        ## Print and save coherence values to file
        with open(self.fns["performance"],'a+') as perf:
            print("*** Coherence ***",file=perf)
            print("Umass coherence (using "+str(self.training_fraction)+" percent of documents as training set): "+str(umass_train_value))
            print("Umass coherence (using "+str(self.training_fraction)+" percent of documents as training set): "+str(umass_train_value),file=perf)

        return umass_train_value

    ### function for creating an enhanced version of the dataset with year and comment indices (used in topic contribution and theta calculation)
    def Get_Indexed_Dataset(self):
        with open(self.fns["lda_prep"],'r') as f:
            indexed_dataset = [] # initialize the full dataset

            if not self.all_:
                assert Path(self.fns["random_indices"]).is_file()
                with open(self.fns["random_indices"]) as g:
                    rand_subsample = []
                    for line in g:
                        if line.strip() != "":
                            rand_subsample.append(int(line))

            per, cumulative=Get_Counts(frequency=self.topic_cont_freq)

            # Only start the counter where counts are greater than 0
            cumulative_as_arr=np.array(cumulative)
            cumulative_mask=np.ma.masked_where(cumulative_as_arr>0, cumulative_as_arr).mask
            counter=list(cumulative_mask).index(True)
            for comm_index,comment in enumerate(f): # for each comment
                if comm_index >= cumulative[counter]:
                    counter += 1 # update the counter if need be

                if self.all_ or (not self.all_ and comm_index in rand_subsample):
                        indexed_dataset.append((comm_index,comment,counter))
                    # append the comment index, the text and the relevant month
                    # or year to the dataset

        return indexed_dataset

    ### Topic Contribution (threaded) ###
    ### Define a function that retrieves the most likely topic for each word in
    # a comment and calculates
    def Topic_Asgmt_Retriever_Multi(self, indexed_comment):
        ## initialize needed vectors
        dxt = np.zeros([num_topics,1]) # a vector for the normalized
        # contribution of each topic to the comment
        analyzed_comment_length = 0 # a counter for the number of words in a
        # comment for which the model has predictions

        # create a list to write index, month and year, topic assignments to a
        # CSV file for analysis in R
        data_for_R = [indexed_comment[0],indexed_comment[2],[]]
        if self.topic_cont_freq == "monthly":
            data_for_R.append(2006+floor(float(indexed_comment[2]) / float(12)))

        comment=indexed_comment[1].strip().split()
        bow=self.dictionary.doc2bow(comment)
        # get per-word topic probabilities for the document
        gamma, phis = self.ldamodel.inference([bow], collect_sstats=True)

        for word_id, freq in bow: # iterate over the word-topic assignments
            try:
                phi_values=[ phis[i][word_id] for i in range(self.num_topics) ]
            except KeyError:
                # Make sure the word either has a probability assigned to all
                # topics, or to no topics
                assert all([ word_id not in phis[i] for i in range(self.num_topics) ])
                continue

            # retrieve the phi values for various words in the comment
            topic_asgmts=sorted(enumerate(phi_values), key=lambda x:x[1],
                                reverse=True)
            # add the most probable topic to the list to be written to a CSV
            data_for_R[2].append(topic_asgmts[0][0])

            if self.one_hot:
                dxt[topic_asgmts[0][0],0] += freq
            else:
                assert len(phi_values)==self.num_topics
                assert essentially_eq(sum(phi_values), freq)
                for topic, phi_val in enumerate(phi_values):
                    dxt[topic,0] += phi_val
            analyzed_comment_length += freq # update word counter

        if analyzed_comment_length > 0: # if the model had predictions for at
        # least some of the words in the comment
            # normalize the topic contribution using comment length
            dxt = (float(1) / float(analyzed_comment_length)) * dxt
            # update the vector of topic contributions
            Running_Sums[indexed_comment[2]].Update_Val(dxt)

        else: # if the model had no reasonable topic proposal for any of the words in the comment
            no_predictions[indexed_comment[2]].Increment() # update the no_predictions counter

        return data_for_R

    ### Define the main function for multi-core calculation of topic contributions
    def Topic_Contribution_Multicore(self):
        # timer
        print("Started calculating topic contribution at " + time.strftime('%l:%M%p'))

        ## check for the existence of the preprocessed dataset
        if not Path(self.fns["lda_prep"]).is_file():
            raise Exception('The preprocessed data could not be found')

        ## initialize shared vectors for yearly topic contributions
        global Running_Sums
        Running_Sums = {}

        _, cumulative=Get_Counts(frequency=self.topic_cont_freq)
        per, _=Get_Counts(random=not self.all_, frequency=self.topic_cont_freq)
        no_intervals=len(cumulative)

        ## Create shared counters for comments for which the model has no
        # reasonable prediction whatsoever
        global no_predictions
        no_predictions = {}

        for i in range(no_intervals):
            Running_Sums[i] = Shared_Contribution_Array(self.num_topics)
            no_predictions[i] = Shared_Counter(initval=0)

        ## read and index comments
        indexed_dataset = self.Get_Indexed_Dataset()

        ## call the multiprocessing function on the dataset
        pool = multiprocessing.Pool(processes=CpuInfo())
        inputs=[ (indexed_comment, self.__dict__) for indexed_comment in
                 indexed_dataset ]
        data_for_CSV = pool.map(func=Topic_Asgmt_Retriever_Multi_wrapper, iterable=inputs)
        pool.close()
        pool.join()

        ## Gather topic contribution estimates in one matrix
        output = []
        for i in range(no_intervals):
            output.append(Running_Sums[i].val[:])

        output = np.asarray(output)

        for i in range(no_intervals):
            if np.all(output[i,:]==0):
                continue
            output[i,:] = ( float(1) / (float(per[i]) - no_predictions[i].value )
                          ) * output[i,:]

        np.savetxt(self.fns["topic_cont"], output) # save the topic contribution matrix to file

        ## write data_for_CSV to file
        with open(self.fns["data_for_R"],'a+b') as data: # create the file
            writer_R = csv.writer(data) # initialize the CSV writer
            if len(data_for_CSV[0]) == 4: # if month information included
                writer_R.writerow(['number','month','year','topic_assignments']) # write headers to the CSV file
                for comment in data_for_CSV: # for each comment
                    month = (comment[1]+1) % 12 # find the relevant month of year
                    if month == 0:
                        month = 12
                    writer_R.writerow([comment[0],month,comment[3],comment[2]]) # the indexing should be in line with the other output files
            elif len(data_for_CSV[0]) == 3: # if only year info is included
                writer1.writerow(['number','year','topic_assignments']) # write headers to the CSV file
                for comment in data_for_CSV:
                    writer_R.writerow([comment[0],comment[1],comment[2]])
            else:
                raise Exception('The topic assignments are not formatted properly.')

        # timer
        print("Finished calculating topic contributions at "+time.strftime('%l:%M%p'))

        return output, indexed_dataset

    ### Function that checks for a topic contribution matrix on file and calls for its calculation if there is none
    def Get_Topic_Contribution(self):
        # check to see if topic contributions have already been calculated
        if not Path(self.fns["topic_cont"]).is_file(): # if not
            # calculate the contributions
            yr_topic_cont, indexed_dataset = self.Topic_Contribution_Multicore()
            np.savetxt(self.fns["topic_cont"], yr_topic_cont) # save the topic contribution matrix to file

            self.indexed_dataset=indexed_dataset
            self.yr_topic_cont=yr_topic_cont

        else: # if there are records on file
            # ask if the contributions should be loaded or calculated again
            Q = raw_input('Topic contribution estimations were found on file. Do you wish to delete them and calculate contributions again? [Y/N]')

            if Q == 'Y' or Q == 'y': # re-calculate
                # calculate the contributions
                yr_topic_cont, indexed_dataset = self.Topic_Contribution_Multicore()
                np.savetxt(self.fns["topic_cont"], yr_topic_cont) # save the topic contribution matrix to file

                self.indexed_dataset=indexed_dataset
                self.yr_topic_cont=yr_topic_cont

            elif Q == 'N' or Q == 'n': # load from file
                print("Loading topic contributions and indexed dataset from file")

                indexed_dataset = self.Get_Indexed_Dataset()
                yr_topic_cont = np.loadtxt(self.fns["topic_cont"])

                self.indexed_dataset=indexed_dataset
                self.yr_topic_cont=yr_topic_cont

            else: # if the answer is neither yes, nor no
                print("Operation aborted. Please note that loaded topic contribution matrix and indexed dataset are required for determining top topics and sampling comments.")
                pass

    # Determine the top topics based on average per-comment contribution over time
    def get_top_topics(self, yr_topic_cont=None):
        if isinstance(yr_topic_cont, type(None)):
            yr_topic_cont=self.yr_topic_cont
        # initialize a vector for average topic contribution

        # get the comment count for each month
        per, cumulative = Get_Counts(frequency=self.topic_cont_freq)

        # scale contribution based on the number of comments in each month
        scaled = np.zeros_like(yr_topic_cont)
        for i in range(len(per)):
            if per[i] != 0:
                scaled[i,:] = (float(per[i]) / float(cumulative[-1])) * yr_topic_cont[i,:]

        # average contribution for each topic
        avg_cont = np.empty(self.num_topics)
        for i in range(self.num_topics):
            # avg_cont[i] = np.sum(scaled[:,i])
            avg_cont[i] = np.mean(yr_topic_cont[24:,i])

        # Find the indices of the [sample_topics] fraction of topics that have
        # the greatest average contribution to the model
        if not isinstance(sample_topics,type(None)): # if based on fraction
            top_topic_no = int(ceil(self.sample_topics*self.num_topics))
            self.top_topics = avg_cont.argsort()[-top_topic_no:][::-1]
        # or the indices of topics that have passed the threshold for
        # consideration based on average contribution
        else: # if based on threshold
            self.top_topics = np.where(avg_cont >= top_topic_thresh)[0]

        # If a set of top topics is provided beforehand
        if not isinstance(self.top_topic_set, type(None)):
            self.top_topics = self.top_topic_set

        print(self.top_topics)

    ### Define a function for plotting the temporal trends in the top topics
    def Plotter(self, name):
        plotter = []
        for topic in self.top_topics:
            plotter.append(self.yr_topic_cont[24:,topic].tolist())

        plots = {}
        for i in range(len(self.top_topics.tolist())):
            plots[i]= plt.plot(range(1,len(plotter[0])+1),plotter[i],
                               label='Topic '+str(self.top_topics[i]))
        plt.legend(loc='best')
        plt.xlabel('{}/{}-{}/{}'.format(dates[24][1], dates[24][0], dates[-1][1],
                                        dates[-1][0]))
        plt.ylabel('Topic Probability')
        plt.title('Contribution of the top topics to the LDA model for {}/{}-{}/{}'.format(
                  dates[24][1], dates[24][0], dates[-1][1], dates[-1][0]))
        plt.grid(True)
        plt.savefig(name)
        plt.show()

    ### Function for multi-core processing of comment-top topic probabilities
    ### IDEA: Add functionality for choosing a certain year (or interval) for which we ask the program to sample comments. Should be simple (indexed_dataset[2])
    def Top_Topics_Theta_Multicore(self):
        # timer
        print("Started calculating theta at " + time.strftime('%l:%M%p'))

        ## filter dataset comments based on length and create a BOW for each comment
        dataset = [] # initialize dataset

        for document in self.indexed_dataset: # for each comment in the
            # indexed_dataset
            if self.min_comm_length == None: # if not filtering based on comment
                # length add a tuple including comment index, bag of words
                # representation and relevant time interval to the dataset
                dataset.append((document[0],
                    self.dictionary.doc2bow(document[1].strip().split()),
                    document[2]))

            else: # if filtering based on comment length
                if len(document[1].strip().split()) > self.min_comm_length:
                    # filter out short comments
                    # add a tuple including comment index, bag of words
                    # representation and relevant time interval to the dataset
                    dataset.append((document[0],
                        self.dictionary.doc2bow(document[1].strip().split()),
                        document[2]))

        ## call the multiprocessing function on the dataset
        theta_with_none = theta_func(dataset, self.ldamodel, self.top_topics)

        ## flatten the list and get rid of 'None's
        theta = []
        for comment in theta_with_none:
            if comment is not None:
                for item in comment:
                    theta.append(item)

        return theta

    ### Function that calls for calculating, re-calculating or loading theta estimations for top topics
    def Get_Top_Topic_Theta(self):
        # check to see if theta for top topics has already been calculated
        if not Path(self.fns["theta"]).is_file(): # if not
            theta = self.Top_Topics_Theta_Multicore() # calculate theta

            # save theta to file
            with open(self.fns["theta"],'a+') as f:
                for element in theta:
                    f.write(' '.join(str(number) for number in element) + '\n')

            self.theta=theta

        else: # if there are records on file
            # ask if theta should be loaded or calculated again
            Q = raw_input('Theta estimations were found on file. Do you wish to delete them and calculate probabilities again? [Y/N]')

            if Q == 'Y' or Q == 'y': # re-calculate
                os.remove(self.fns["theta"]) # delete the old records

                theta = self.Top_Topics_Theta_Multicore() # calculate theta

                # save theta to file
                with open(self.fns["theta"],'a+') as f:
                    for element in theta:
                        f.write(' '.join(str(number) for number in element) + '\n')

                self.theta=theta

            elif Q == 'N' or Q == 'n': # load from file
                print("Loading theta from file")

                with open(self.fns["theta"],'r') as f:
                    theta = [tuple(map(float, number.split())) for number in f]

                self.theta=theta

            else: # if the answer is neither yes, nor no
                print("Operation aborted. Please note that loaded theta is required for sampling top comments.")

    ### Defines a function for finding the [sample_comments] most representative length-filtered comments for each top topic
    def Top_Comment_Indices(self):
        top_topic_probs = {} # initialize a dictionary for all top comment indices
        sampled_indices = {} # initialize a dictionary for storing sampled comment indices
        sampled_probs = {} # initialize a list for storing top topic contribution to sampled comments
        sampled_ids = {} # intialize a list for storing randomly chosen six-digit IDs for sampled comments

        for topic in self.top_topics: # for each top topic
            # find all comments with significant contribution from that topic
            top_topic_probs[topic] = [element for element in self.theta if
                                      element[1] == topic]
            top_topic_probs[topic] = sorted(top_topic_probs[topic], key=lambda x: x[2],reverse=True) # sort them based on topic contribution

            # find the [sample_comments] comments for each top topic that show the greatest contribution
            sampled_indices[topic] = []
            sampled_probs[topic] = []
            for element in top_topic_probs[topic][:min(len(top_topic_probs[topic]),sample_comments)]:
                sampled_indices[topic].append(element[0]) # record the index
                sampled_probs[topic].append(element[2]) # record the contribution of the topic
                # suggest a random 8-digit id for the sampled comment
                prop_id = np.random.random_integers(low=10000000,high=99999999)
                while prop_id in sampled_ids: # resample in the unlikely event the id is already in use
                    prop_id = np.random.random_integers(low=10000000,high=99999999)
                sampled_ids[element[0]] = prop_id # store the random id
        return sampled_indices,sampled_probs,sampled_ids

    ### retrieve the original text of sampled comments and write them to file along with the relevant topic ID
    # IDEA: Should add the possibility of sampling from specific year(s)
    def Get_Top_Comments(self):
        # timer
        print("Started sampling top comments at " + time.strftime('%l:%M%p'))

        # find the top comments associated with each top topic
        sampled_indices,sampled_probs,sampled_ids = self.Top_Comment_Indices()

        _, yearly_cumulative = Get_Counts(frequency="yearly")
        _, monthly_cumulative = Get_Counts(frequency="monthly")

        if not Path(self.fns["original_comm"]).is_file(): # if the original relevant comments are not already available on disk, read them from the original compressed files
            # json parser
            decoder = json.JSONDecoder(encoding='utf-8')

            # check for the presence of data files
            if not glob.glob(self.path+'/*.bz2'):
                raise Exception('No data file found')

            # open two CSV files for recording sampled comment keys and ratings
            with open(self.fns["sample_keys"],'a+b') as sample_keys, \
            open(self.fns["sample_ratings"],'a+b') as sample_ratings, \
            open(self.fns["sampled_comments"],'a+') as fout: # determine the I/O files:

                ## iterate over files in directory to find the relevant documents
                counter = 0 # counting the number of all processed comments
                year_counter = 0 # the first year in the corpus (2006)

                # create CSV file for recording sampled comment information
                writer_keys = csv.writer(sample_keys) # initialize the CSV writer
                writer_keys.writerow(['comment number','random index','month','year','topic','contribution']) # write headers to the CSV file

                # create CSV file for recording sampled comment ratings
                writer_ratings = csv.writer(sample_ratings)
                writer_ratings.writerow(['index','pro','values','consequences','preferences','interpretability'])

                bag_of_comments = {} # initialize a list for the sampled
                # original comments. Will be used to shuffle comments before
                # writing them to file

                # iterate through the files in the 'path' directory in alphabetic order
                for filename in sorted(os.listdir(self.path)):
                    # only include relevant files
                    if os.path.splitext(filename)[1] == '.bz2' and 'RC_' in filename:
                        ## prepare files
                        # open the file as a text file, in utf8 encoding
                        with bz2.BZ2File(filename,'r') as fin:

                            ## read data
                            for line in fin: # for each comment
                                # parse the json, and turn it into regular text
                                comment = decoder.decode(line)
                                original_body = HTMLParser.HTMLParser().unescape(comment["body"]) # remove HTML characters

                                # filter comments by relevance to the topic
                                if len(GAYMAR.findall(original_body)) > 0 or len(MAREQU.findall(original_body)) > 0:
                                    # clean the text for LDA
                                    body = Parser(stop=self.stop).LDA_clean(original_body)

                                    # if the comment body is not empty after preprocessing
                                    if body.strip() != "":
                                        counter += 1 # update the counter

                                        # update year counter if need be
                                        if counter-1 >= yearly_cumulative[year_counter]:
                                            year_counter += 1

                                        # find the relevant month
                                        for value in monthly_cumulative:
                                            if value > counter-1:
                                                month = (monthly_cumulative.index(value) + 1) % 12
                                                if month == 0:
                                                    month = 12
                                                break

                                        for topic,indices in sampled_indices.iteritems():
                                            if counter-1 in indices:
                                                # remove mid-comment lines and set encoding
                                                original_body = original_body.replace("\n","")
                                                original_body = original_body.encode("utf-8")

                                                bag_of_comments[sampled_ids[counter-1]] = " ".join(comment.strip().split())

                                                # print the values to CSV file
                                                itemindex = sampled_indices[topic].index(comm_index)
                                                writer_keys.writerow([counter-1,sampled_ids[counter-1],month,2006+year_counter,topic,sampled_probs[topic][itemindex]])

                                                # print the comment to file
                                                print(" ".join(original_body.strip().split()),file=fout)

                                                break # if you found the index in one of the topics, no reason to keep looking

                ## shuffle the comments and write to file for human raters
                random_keys = np.random.permutation(bag_of_comments.keys())
                for random_key in random_keys:
                    writer_ratings.writerow([random_key])
                    print('index: '+str(random_key),file=fout)
                    print(" ".join(bag_of_comments[random_key].strip().split()),file=fout)

            # timer
            print("Finished sampling top comments at " + time.strftime('%l:%M%p'))

        else: # if a file containing only the original relevant comments is available on disk
            with open(self.fns["original_comm"],'r') as fin, \
                 open(self.fns["sample_keys"],'a+b') as sample_keys, \
                 open(self.fns["sample_ratings"],'a+b') as sample_ratings, \
                 open(self.fns["sampled_comments"],'a+') as fout: # determine the I/O files

                # create CSV file for recording sampled comment information
                writer_keys = csv.writer(sample_keys) # initialize the CSV writer
                writer_keys.writerow(['comment number','random index','month','year','topic','contribution']) # write headers to the CSV file

                # create CSV file for recording sampled comment ratings
                writer_ratings = csv.writer(sample_ratings)
                writer_ratings.writerow(['index','pro','values','consequences','preferences','interpretability'])

                bag_of_comments = {} # initialize a list for the sampled
                # original comments. Will be used to shuffle comments before
                # writing them to file
                year_counter = 0 # initialize a counter for the comment's year

                for comm_index,comment in enumerate(fin): # iterate over the original comments

                    # update year counter if need be
                    if comm_index >= yearly_cumulative[year_counter]:
                        year_counter += 1

                    for topic,indices in sampled_indices.iteritems():
                        if comm_index in indices:

                            # find the relevant month
                            for value in monthly_cumulative:
                                if value > comm_index:
                                    month = (monthly_cumulative.index(value) + 1) % 12
                                    if month == 0:
                                        month = 12
                                    break

                            bag_of_comments[sampled_ids[comm_index]] = " ".join(comment.strip().split())

                            # print the values to CSV file
                            itemindex = sampled_indices[topic].index(comm_index)
                            writer_keys.writerow([comm_index,sampled_ids[comm_index],month,2006+year_counter,topic,sampled_probs[topic][itemindex]])

                            break # if you found the index in one of the topics, no reason to keep looking

                ## shuffle the comments and write to file for human raters
                random_keys = np.random.permutation(bag_of_comments.keys())
                for random_key in random_keys:
                    writer_ratings.writerow([random_key])
                    print('index: '+str(random_key),file=fout)
                    print(" ".join(bag_of_comments[random_key].strip().split()),file=fout)

                # timer
                print("Finished sampling top comments at " + time.strftime('%l:%M%p'))

    ## function for sampling the most impactful comments
    def sample_pop(self,num_pop=num_pop,min_comm_length=min_comm_length):

        assert Path(self.fns["counts"]).is_file()
        assert Path(self.fns["original_comm"]).is_file()
        assert Path(self.fns["votes"]).is_file()

        # Retrieve the list of upvote/downvote values
        with open(self.fns["counts"],'r') as f:
            timelist = []
            for line in f:
                if line.strip() != "":
                    timelist.append(int(line))

        # Retrieve the text of the original comments
        with open("original_comm",'r') as f:
            orig_comm = []
            for line in f:
                if line.strip() != "":
                    orig_comm.append(line.strip())

        # Retrieve the pre-processed comments
        with open("lda_prep",'r') as f:
            lda_prep = []
            for line in f:
                if line.strip() != "":
                    lda_prep.append(line.strip())

        # Calculate the relevant month and year for each comment
        rel_month = np.zeros((timelist[-1],1),dtype="int32")
        month_of_year = np.zeros((timelist[-1],1),dtype="int32")
        rel_year = np.zeros((timelist[-1],1),dtype="int32")
        for rel_ind in range(timelist[-1]):
            # Find the relevant month and year
            rel_month[rel_ind,0] = next((i+1 for i in range(len(timelist)) if timelist[i] > rel_ind),141)
            month_of_year[rel_ind,0] = rel_month[rel_ind,0] % 12
            if month_of_year[rel_ind,0] == 0:
                month_of_year[rel_ind,0] = 12
            rel_year[rel_ind,0] = int(2006+int(floor(rel_month[rel_ind,0]/12)))

        # retrieve comment scores from file
        with open("votes",'a+') as f:
            vote_count = dict()
            abs_vote_count = dict()
            for number,line in enumerate(f):
                if line.strip() != "":
                    vote_count[str(number)] = int(line)
                    abs_vote_count[str(number)] = abs(int(line))

        # sort scores based on absolute value
        sorted_votes = sorted(abs_vote_count.items(), key=operator.itemgetter(1),reverse=True)

        # Find the num_pop comments with the highest impact on the discourse
        counter = 0
        results = []
        abs_results = []
        popular = []
        for x in sorted_votes:
            comment = orig_comm[int(x[0])]
            if min_comm_length != None:
                if len(comment.strip().split()) > min_comm_length:
                    counter += 1
                    results.append(vote_count[x[0]])
                    abs_results.append(x[1])
                    popular.append(int(x[0]))
            else:
                counter += 1
                results.append(vote_count[x[0]])
                abs_results.append(x[1])
                popular.append(int(x[0]))
            if counter == num_pop:
                break

        # retrieve the text of popular comments
        pop_orig_texts = []
        pop_proc_texts = []
        for pop_comment in popular:
            pop_orig_texts.append(orig_comm[pop_comment])
        for pop_comment in popular:
            pop_proc_texts.append(lda_prep[pop_comment])

        # Retrieve topic probability from the model for each popular comment
        pop_comm_topic = []
        pop_comm_contrib = []
        for comment in pop_proc_texts:
            comment= comment.strip().split()
            bow = self.dictionary.doc2bow(comment)
            # get topic probabilities for the document
            topic_dist = self.ldamodel.get_document_topics([bow], per_word_topics=False)
            # sort the probabilities
            sorted_topics = sorted(topic_dist[0], key=lambda x:x[1],reverse=True)
            # add the most probable topic to the list to be written to a CSV
            # pop_comm_topic.append(sorted_topics[0][0])
            # pop_comm_contrib.append(sorted_topics[0][1])
            pop_comm_topic.append(sorted_topics) # this appends ALL contributing topics

        # write the most popular comments, their timeframe and score to file
        with open(self.fns["popular_comments"],'w+b') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['text','year','month','score','topic','contribution'])
            writer.writerow(['text','year','month','score','topic contribution'])
            for number,pop_comment in enumerate(popular):
                # writer.writerow([pop_orig_texts[number],rel_year[pop_comment,0],
                # month_of_year[pop_comment,0],results[number],
                # pop_comm_topic[number],pop_comm_contrib[number]])
                writer.writerow([pop_orig_texts[number],rel_year[pop_comment,0],
                month_of_year[pop_comment,0],results[number],
                pop_comm_topic[number]])

class NNModel(ModelEstimator):
    def __init__(self, FrequencyFilter=FrequencyFilter, **kwargs):
        ModelEstimator.__init__(self, **kwargs)
        self.FrequencyFilter=FrequencyFilter
        self.set_key_list = ['train','dev','test'] # for NN
        self.sets    = {key: [] for key in self.set_key_list} # for NN
        self.indexes = {key: [] for key in self.set_key_list}
        self.lengths = {key: [] for key in self.set_key_list}
        self.Max     = {key: [] for key in self.set_key_list}
        self.vote     = {key: [] for key in self.set_key_list} # for NN

        self.fns=self.get_fns()

    def get_fns(self, **kwargs):
        fns={ "nn_prep":"{}/nn_prep".format(self.path),
              "counts": parser_fns["counts"] if self.all_ else parser_fns["counts_random"],
              "dictionary":"{}/dict_{}".format(self.path, self.regression),
              "train_set": "{}/train_{}".format(self.path, self.regression),
              "dev_set":"{}/dev_{}".format(self.path, self.regression),
              "test_set":"{}/test_{}".format(self.path, self.regression),
              "indexed_train_set": "{}/indexed_train_{}".format(self.path, self.regression),
              "indexed_dev_set":"{}/indexed_dev_{}".format(self.path, self.regression),
              "indexed_test_set":"{}/indexed_test_{}".format(self.path, self.regression),
              "sample_ratings": "{}/sample_ratings.csv".format(self.output_path)
            }
        for k, v in kwargs.items():
            fns[k]=v
        return fns

    ### load or create vocabulary and load or create indexed versions of comments in sets
    # NOTE: Only for NN. For LDA we use gensim's dictionary functions
    def Index_Set(self, set_key):
        ## record word frequency in the entire dataset
        frequency = defaultdict(int)
        if Path(self.fns["nn_prep"]).is_file(): # look for preprocessed data
            fin = open(self.fns["nn_prep"],'r')
            for comment in fin: # for each comment
                for token in comment.split(): # for each word
                    frequency[token] += 1 # count the number of occurrences

        else: # if no data is found, raise an error
            raise Exception('Pre-processed dataset could not be found')

        # if indexed comments are available and we are trying to index the training set
        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file() and set_key == 'train':
            # If the vocabulary is available, load it
            if Path(self.path+"/dict_"+str(self.regression)).is_file():
                print("Loading dictionary from file")

                with open(self.fns["dictionary"],'r') as f:
                    for line in f:
                        if line.strip() != "":
                            (key, val) = line.split()
                            V[key] = int(val)

            else: # if the vocabulary is not available
                # delete the possible dictionary-less indexed training set file
                if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                    os.remove(self.fns["indexed_{}_set".format(set_key)])

        # if indexed comments are available, load them
        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
            print("Loading the set from file")

            with open(self.fns["indexed_{}_set".format(set_key)],'r') as f:
                for line in f:
                    assert line.strip() != ""
                    comment = []
                    for index in line.split():
                        comment.append(index)
                    self.indexes[set_key].append(comment)

        else: # if the indexed comments are not available, create them
            if set_key == 'train': # for training set
                # timer
                print("Started creating the dictionary at " + time.strftime('%l:%M%p'))

                ## initialize the vocabulary with various UNKs
                self.V.update({"*STOP2*":1,"*UNK*":2,"*UNKED*":3,"*UNKS*":4,"*UNKING*":5,"*UNKLY*":6,"*UNKER*":7,"*UNKION*":8,"*UNKAL*":9,"*UNKOUS*":10,"*STOP*":11})

            ## read the dataset and index the relevant comments
            fin.seek(0) # go to the beginning of the data file
            for counter,comm in enumerate(fin): # for each comment
                if counter in self.sets[set_key]: # if it belongs in the set
                    comment = [] # initialize a list

                    for word in comm.split(): # for each word
                        if frequency[word] > FrequencyFilter: # filter non-frequent words
                            if word in self.V.keys(): # if the word is already in the vocabulary
                                comment.append(self.V[word]) # index it and add it to the list

                            elif set_key == 'train': # if the word is not in vocabulary and we are indexing the training set
                                    if len(self.V)-11 <= self.MaxVocab: # if the vocabulary still has room (not counting STOPs and UNKs)
                                        self.V[word] = len(self.V)+1 # give it an index (leave index 0 for padding)
                                        comment.append(self.V[word]) # append it to the list of words

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

                    self.indexes[set_key].append(comment) # add the comment to the indexed list

            ## save the vocabulary to file
            if set_key == 'train':
                vocab = open(self.fns["dictionary"],'a+')
                for word,index in self.V.iteritems():
                    print(word+" "+str(index),file=vocab)
                vocab.close

            ## save the indexed datasets to file
            with open(self.fns["indexed_{}_set".format(set_key)],'a+') as f:
                for comment in self.indexes[set_key]:
                    assert len(comment) != 0
                    for ind,word in enumerate(comment):
                        if ind != len(comment) - 1:
                            print(word,end=" ",file=f)
                        elif ind == len(comment) - 1:
                            print(word,file=f)

            # ensure that datasets have the right size
            assert len(self.indexes[set_key]) == len(self.sets[set_key])

        # timer
        print("Finished indexing the "+set_key+" set at " + time.strftime('%l:%M%p'))
