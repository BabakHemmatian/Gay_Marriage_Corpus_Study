from __future__ import division
"""TODO: Add docstring.
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from config import *
from reddit_parser import Parser

class Model(object):
    def __init__(self, training_data = None, test_data = None):
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
        # Instantiate a Parser object just for its _clean method
        self.parser = Parser()

    def train(self):
        doc_cons = self.training_data.loc[self.training_data["category"] == 1]["text"]
        doc_vb = self.training_data.loc[self.training_data["category"] == 0]["text"]
        #assert len(doc_cons) + len(doc_vb) == len(self.training_data)
        doc_cons = " ".join(doc_cons.values)
        doc_vb = " ".join(doc_vb.values)

        # Clean documents
        doc_cons = self.parser._clean(doc_cons)
        doc_vb = self.parser._clean(doc_vb)

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

    # Categorize a string as consequentialist or values-based by the sum of the
    # weights assigned to its component tokens. Under a naive Bayesian model of
    # speech generation (all tokens are independently sampled), this
    # corresponds to the log odds that every token in the string was sampled
    # from the consequentialist corpus
    def _categorize_comment(self, text):
        text = self.parser._clean(text)
        text = text.split()
        text = [ token for token in text if token in self.features ]
        ixs = [ list(self.features).index(word) for word in text ]
        score = np.sum(self.wghts[ixs])
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
    k, dat = args

    accuracy = []
    folds = np.array_split(dat.index, k)
    for fold in folds:
        training_data = dat.loc[[ix for ix in dat.index if ix not in fold]]
        test_data = dat.loc[fold]
        assert round(len(training_data) / len(test_data)) == k - 1
        mod = Model(training_data = training_data, test_data = test_data)
        mod.train()
        accuracy.append(mod.score()[0])
    return np.mean(accuracy)

def repeat_kfold(n = 10, k = 10, n_cpus = mp.cpu_count() - 1):
    dat = pickle.load(open(ALL_FN, "rb"))
    pool = mp.Pool(n_cpus)
    accuracy = pool.map(kfold, [(k, dat)] * n)
    return np.mean(accuracy)
