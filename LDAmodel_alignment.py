from __future__ import division

"""Choosing different samples of documents to train an LDA model can lead to
different resulting distribution of topics. This module will compute the model
alignment distance and topic overlap for two arbitrary models of the gensim
models.ldamodel.LdaModel class. Model alignment metrics presented in Murdock,
 Jaimie, Zeng, Jiaan, and Allen, Colin. (2016). Towards Evaluation of
Cultural-scale Claims in Light of Topic Model Sampling Effects, International
Conference on Computational Social Science, June 23-26.
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

    # "...we perform a topic alignment between each pair of models by computing
    # the Jensen-Shannon distance (JSD) between the word probability
    # distributions for each topic in M1 and M2. Each topic in M1 is matched to
    # the closest topic in M2, allowing for multiple topics in M1 to be aligned
    # to the same topic in M2." (Murdock et. al. 2016, pg. 2)
    def get_alignment_pairs(self):       
        self.alignment_pairs=dict()
        self.alignment_distances=np.full((self.m1.num_topics, self.m2.num_topics
                                         ), np.nan)

        # Create term topic matrices aligned on words
        t2t_lu=defaultdict(lambda:None)
        m1_word2id=dict( (v, k) for k, v in self.m1.id2word.iteritems() )
        m2_word2id=dict( (v, k) for k, v in self.m2.id2word.iteritems() )
        topics1=self.m1.get_topics()
        topics2_=self.m2.get_topics()
        topics2=[]
        for i in range(len(topics1)):
            topic=topics1[i]
            topic_=[]
            for j in range(len(topic)):
                try:
                    j_=m2_word2id[self.m1.id2word[j]]
                    topic_.append(topics2_[i][j_])
                except KeyError:
                    topic_.append(0)
            topics2.append(topic_)

        # Create alignment pairs. For every topic in M1, create a pair with the
        # topic in M2 
        for i in range(len(topics1)):
            _jsds=defaultdict(list)
            for j in range(len(topics2)):
                aprob=thoth.prob_from_array(topics1[i])
                bprob=thoth.prob_from_array(topics2[j])
                _jsds[thoth.jsd(aprob, bprob, .5)].append(j)
            j, ts=min(_jsds.keys()), _jsds[min(_jsds.keys())]
            # If multiple topics in M2 have the same JSD wrt a given topic in
            # M1, choose one randomly to complete the alignment pair.
            t=np.random.choice(ts)
            self.alignment_pairs[i]=t
            self.alignment_distances[i][t]=j
            self.alignment_distances[t][i]=j

    # "The alignment distance is the average JSD of each alignment pair."
    # (Murdock et. al. 2016, pg. 2)
    def get_alignment_distance(self):
        self.alignment_distance=np.mean(self.alignment_distances[~np.isnan(self.alignment_distances)])
        
    # The topic overlap is the percentage of topics in M2 that were selected as
    # the nearest neighbor of a topic in M1." (Murdock et. al. 2016, pg. 2)
    def get_topic_overlap(self):
        self.topic_overlap=( len(set(self.alignment_pairs.values()))/
                             self.m2.num_topics )
