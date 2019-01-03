from __future__ import division
"""TODO: Add docstring.
"""

from collections import defaultdict
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
import string
from config import *
from reddit_parser import Parser

class Model(object):
    def __init__(self, throwaway = False, **kwargs):
        if not throwaway:
            self.init(**kwargs)

    def init(self, training_data = None, test_data = None):
        assert ( isinstance(training_data, pd.DataFrame) or
                 isinstance(training_data, type(None)) )
        assert ( isinstance(test_data, pd.DataFrame) or
                 isinstance(test_data, type(None)) )
        if isinstance(training_data, type(None)):
            # Load training data
            training_data = pickle.load(open(TRAINING_FN, "rb"))
        if isinstance(test_data, type(None)):
            # Load test data
            test_data = pickle.load(open(TEST_FN, "rb"))
        self.training_data = training_data
        self.test_data = test_data
        # Instantiate a Parser object just for its LDA_clean method
        self.parser = Parser()

        self.id_ = "".join(np.random.choice(list(string.lowercase), 4))

    def train(self, save = False):
        doc_cons = self.training_data.loc[self.training_data["category"] == 1]["text"]
        doc_vb = self.training_data.loc[self.training_data["category"] == 0]["text"]
        #assert len(doc_cons) + len(doc_vb) == len(self.training_data)
        doc_cons = " ".join(doc_cons.values)
        doc_vb = " ".join(doc_vb.values)

        # Clean documents
        doc_cons = self.parser.LDA_clean(doc_cons)
        doc_vb = self.parser.LDA_clean(doc_vb)

        # Get the frequency of occurence of each word in each corpus
        vectorizer = CountVectorizer()
        freqs = vectorizer.fit_transform([doc_cons, doc_vb]).toarray()
        # L1 smoothing to deal with words not seen in some corpus
        freqs += 1
        # Get the log-odds that every word appears in the set of
        # consequentialist comments
        _cfreqs = freqs[0]
        _cprobs = _cfreqs/np.sum(_cfreqs)
        np.testing.assert_approx_equal(np.sum(_cprobs), 1)
        _vfreqs = freqs[1]
        _vprobs = _vfreqs/np.sum(_vfreqs)
        np.testing.assert_approx_equal(np.sum(_vprobs), 1)
        self.wghts = np.log(_cprobs/_vprobs)
        np.testing.assert_allclose(self.wghts, -1 * np.log(_vprobs/_cprobs))
        self.features = np.array(vectorizer.get_feature_names())

        if save:
            with open("weights-" + self.id_, "w") as wfh:
                wfh.write(" ".join(map(str, self.wghts)))
            with open("features-" + self.id_, "w") as wfh:
                wfh.write(" ".join(self.features))

    def _get_score(self, text, process = True):
        if process:
            text = self.parser.LDA_clean(text)
            text = np.array(text.split())
        text = text[np.in1d(text, self.features)]
        ixs = [ np.where(self.features == word)[0][0] for word in text ]
        score = np.sum(self.wghts[ixs])
        return score

    # Categorize a string as consequentialist or values-based by the sum of the
    # weights assigned to its component tokens. Under a naive Bayesian model of
    # speech generation (all tokens are independently sampled), this
    # corresponds to the log odds that every token in the string was sampled
    # from the consequentialist corpus
    def _categorize_comment(self, text, **kwargs):
        score = self._get_score(text, **kwargs)
        # If the model assigns a score of 0 to a comment, it might make sense to
        # penalize it, since it's not able to categorize that comment. We want
        # to give the Naive Bayes a fair shot w.r.t. the LDA though, so just
        # exclude that judgment.
        if score == 0:
            return np.nan, score
        cat = 1 if score > 0 else 0
        return cat, score

    def score(self):
        nright, n, nindifferent = 0, 0, 0
        # Babak tested the LDA model on the basis of mean rating, not category
        ##for cat, text in self.test_data[["category", "text"]].values:
        ##    assert cat in (0,1)
        test_data_ = self.test_data.loc[self.test_data["mean"] != 4]
        for avgr, text in test_data_[["mean", "text"]].values:
            if avgr == 4:
                continue
            cat = 1 if avgr > 4 else 0
            cat_, score_ = self._categorize_comment(text)
            if np.isnan(cat_):
                nindifferent += 1
                continue
            if cat_ == cat:
                nright += 1
            n += 1
        return nright / n, nindifferent / len(test_data_)

def kfold(args):
    k, dat, seed = args
    # I'm not sure if every new spawned process uses the same seed, so force
    # them to use different seeds just in case
    np.random.seed(seed)

    accuracy = []
    ixs = dat.index.values
    np.random.shuffle(ixs)
    folds = np.array_split(ixs, k)
    for fold in folds:
        training_data = dat.loc[[ix for ix in dat.index if ix not in fold]]
        test_data = dat.loc[fold]
        assert round(len(training_data) / len(test_data)) == k - 1
        mod = Model(training_data = training_data, test_data = test_data)
        mod.train(save = True)
        accuracy.append(mod.score()[0])
    return np.mean(accuracy)

def repeat_kfold(n = 10, k = 10, n_cpus = mp.cpu_count() - 1):
    dat = pickle.load(open(ALL_FN, "rb"))
    pool = mp.Pool(n_cpus)
    seeds = np.random.randint(1000, size = n)
    accuracy = pool.map(kfold, [(k, dat, seed) for seed in seeds])
    return np.mean(accuracy)

def get_saved_model_params():
    fs = os.listdir(".")
    wfs = [ f for f in fs if re.match("weights-[a-z]{4}", f) ]
    assert len(wfs) == 100
    ffs = [ "features-" + f.split("-")[-1] for f in wfs ]
    return (wfs, ffs)

def get_per_word_wght():
    kwds = defaultdict(int)

    wfs, ffs = get_saved_model_params()
    for wf, ff in zip(wfs, ffs):
        wghts = map(float, open(wf, "r").read().split())
        features = open(ff, "r").read().split()
        feature_wghts = zip(features, wghts)
        for f_, w in feature_wghts:
            kwds[f_] += w
    return kwds

def DEPR_get_temporal_trends():
    cons_words = [ w for w in open("words-cons", "r").read().split("\n") if
                   w.strip() ]
    vb_words = [ w for w in open("words-vb", "r").read().split("\n") if
                 w.strip() ]
    comments = [ l for l in open("lda_prep", "r").read().split("\n") if
                 l.strip() ]
    frac_cons, frac_vb = [], []

    i, comment = 0, comments[0]
    for ix in open("RC_Count_List", "r").read().split("\n"):
        if i >= len(comments) or not ix.strip():
            break
        ix = int(ix)
        n, n_cons, n_vb = 0, 0, 0
        while i < ix:
            text = [ token for token in comment.split() if token.strip() ]
            n += len(text)
            n_cons += len([ token for token in text if token in cons_words ])
            n_vb += len([ token for token in text if token in vb_words ])
            if i+1 >= len(comments):
                i += 1
                break
            i, comment = i+1, comments[i+1]
        if n > 0:
            frac_cons.append(n_cons/n)
            frac_vb.append(n_vb/n)

    return frac_cons, frac_vb

def get_data_for_one_month(args):
    comments, wghts, features = args
    mod = Model(throwaway = True)
    mod.wghts = wghts
    mod.features = features
    n = len(comments)
    all_scores = np.array(map(lambda comment: mod._categorize_comment(comment,
                                                                      process = False),
                          comments))
    if n > 0:
        scores_ = np.array([sum(all_scores[:,1]),
                            list(all_scores[:,0]).count(1) / n,
                            list(all_scores[:,0]).count(0) / n])
    else:
        scores_ = np.array([np.nan, np.nan, np.nan])
    return scores_

def get_temporal_trends(save = True):
    comments = [ l for l in open("lda_prep", "r").read().split("\n") if
                 l.strip() ]
    comments = [ np.array(comment.split()) for comment in comments ]
    ixs = list(map(int, [ l for l in
                          open("RC_Count_List", "r").read().split("\n") if
                          l.strip() ]))
    # Memory errors
    #scores = np.array([0] * 100 * (len(ixs)-1))
    #scores = scores.reshape(((len(ixs)-1), 100))
    #frac_cons = np.array([0] * 100 * (len(ixs)-1))
    #frac_cons = frac_cons.reshape(((len(ixs)-1), 100))
    #frac_vb = np.array([0] * 100 * (len(ixs)-1))
    #frac_vb = frac_vb.reshape(((len(ixs)-1), 100))

    wfs, ffs = get_saved_model_params()
    for i, (wf, ff) in enumerate(zip(wfs, ffs)):
        ##
        id_ = wf.split("-")[-1]
        print ("Sample #" + str(i+1))
        wghts = np.array(list(map(float, open(wf, "r").read().split())))
        features = np.array(open(ff, "r").read().split())
        comments_ = [ comments[ixs[i_-1]:ixs[i_]] for i_ in range(1, len(ixs)) ]
        scores_ = mp.Pool(mp.cpu_count() - 1).map(get_data_for_one_month,
                                                  [ (comments__, wghts, features
                                                    ) for comments__ in 
                                                    comments_ ])
        scores_ = np.array(scores_)
        #scores[:,i] = scores_[:,0]
        #frac_cons[:,i] = scores_[:,1]
        #frac_vb[:,i] = scores_[:,2]

        ##
        if save:
            with open("scores" + id_, "wb") as wfh:
                pickle.dump(scores_, wfh)
            #with open("frac_cons" + id_, "wb") as wfh:
            #    pickle.dump(frac_cons, wfh)
            #with open("frac_vb" + id_, "wb") as wfh:
            #    pickle.dump(frac_vb, wfh)

    #return scores, frac_cons, frac_vb
