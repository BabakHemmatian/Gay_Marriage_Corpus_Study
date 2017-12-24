from __future__ import division

"""Choosing different samples of documents to train an LDA model can lead to
different resulting distribution of topics. This module will compute the model
alignment distance and topic overlap for two arbitrary models of the gensim
models.ldamodel.LdaModel class. Model alignment metrics presented in Murdock,
 Jaimie, Zeng, Jiaan, and Allen, Colin. (2016). Towards Evaluation of
Cultural-scale Claims in Light of Topic Model Sampling Effects, International
Conference on Computational Social Science, June 23-26.

Usage:
> import LDAmodel_alignment as malign
> # To get the JSD between two topics
> # From the same model
> MA=malign.ModelAlignment('sample_lda_models/ldamodel1',
    'sample_lda_models/ldamodel1')
> MA.align_term_topic_matrices()
> MA.get_jsd(MA.topics1[0], MA.topics1[1]) # Same as MA.get_jsd(MA.topics1[0],
    # MA.topics2[1]), because here M1 is M2
0.010366824011974973
>
> # From different models
> MA=malign.ModelAlignment('sample_lda_models/ldamodel1',
    'sample_lda_models/ldamodel2')
> MA.align_term_topic_matrices()
> MA.get_jsd(MA.topics1[0], MA.topics2[0])
0.7051554823023656

> # Compare two models as a whole
> MA.get_alignment_pairs()
> # See JSD between alignment pairs
> MA.alignment_distances
array([[        nan,  0.70096102],
       [ 0.73932173,         nan]])
> # Get the average JSD between closest pairs
> MA.get_alignment_distance()
> MA.alignment_distance
0.72014137605514128
> # Get the proportion of topics in M2 that complete an alignment pair with a
    # topic in M1
> MA.get_topic_overlap()
> MA.topic_overlap
1.0
"""

from collections import defaultdict
from gensim.models.ldamodel import LdaModel
import numpy as np
# thoth-python.org
import thoth.thoth as thoth

class ModelAlignment(object):
    def __init__(self, file1, file2):
        m1=LdaModel.load(file1)
        m2=LdaModel.load(file2)
        if len(m1.id2word.keys())>=len(m2.id2word.keys()):
            self.m1, self.m2=m1, m2
        else:
            self.m1, self.m2=m2, m1

    # Get the Jensen-Shannon distance between two probability distributions.
    def get_jsd(self, a, b):
        aprob=thoth.prob_from_array(a)
        bprob=thoth.prob_from_array(b)
        return thoth.jsd(aprob, bprob, .5)

    # Permute and add 0 entries to the term topic matrix for M2 to align its
    # terms with the terms in the term topic matrix for M1. Creates class
    # attributes topics1 and topics2, which correspond to the term topic
    # matrices for M1 and M2 respectively. Every column in topics1 corresponds
    # to the same word in topics2.
    def align_term_topic_matrices(self):
        t2t_lu=defaultdict(lambda:None)
        m1_word2id=dict( (v, k) for k, v in self.m1.id2word.iteritems() )
        m2_word2id=dict( (v, k) for k, v in self.m2.id2word.iteritems() )
        self.topics1=self.m1.get_topics()
        topics2_=self.m2.get_topics()
        self.topics2=[]
        for i in range(len(self.topics1)):
            topic=self.topics1[i]
            topic_=[]
            for j in range(len(topic)):
                try:
                    j_=m2_word2id[self.m1.id2word[j]]
                    topic_.append(topics2_[i][j_])
                except KeyError:
                    topic_.append(0)
            self.topics2.append(topic_)

    # "...we perform a topic alignment between each pair of models by computing
    # the Jensen-Shannon distance (JSD) between the word probability
    # distributions for each topic in M1 and M2. Each topic in M1 is matched to
    # the closest topic in M2, allowing for multiple topics in M1 to be aligned
    # to the same topic in M2." (Murdock et. al. 2016, pg. 2)
    def get_alignment_pairs(self):       
        self.alignment_pairs=dict()
        self.alignment_distances=np.full((self.m1.num_topics, self.m2.num_topics
                                         ), np.nan)

        if not hasattr(self, 'topics1') or not hasattr(self, 'topics2'):
            self.align_term_topic_matrices()

        # Create alignment pairs. For every topic in M1, create a pair with the
        # topic in M2 
        for i in range(len(self.topics1)):
            _jsds=defaultdict(list)
            for j in range(len(self.topics2)):
                _jsd=self.get_jsd(self.topics1[i], self.topics2[j])
                _jsds[_jsd].append(j)
            j, ts=min(_jsds.keys()), _jsds[min(_jsds.keys())]
            # If multiple topics in M2 have the same JSD wrt a given topic in
            # M1, choose one randomly to complete the alignment pair.
            t=np.random.choice(ts)
            self.alignment_pairs[i]=t
            self.alignment_distances[i][t]=j

    # "The alignment distance is the average JSD of each alignment pair."
    # (Murdock et. al. 2016, pg. 2)
    def get_alignment_distance(self):
        self.alignment_distance=np.mean(self.alignment_distances[~np.isnan(self.alignment_distances)])
        
    # The topic overlap is the percentage of topics in M2 that were selected as
    # the nearest neighbor of a topic in M1." (Murdock et. al. 2016, pg. 2)
    def get_topic_overlap(self):
        self.topic_overlap=( len(set(self.alignment_pairs.values()))/
                             self.m2.num_topics )
