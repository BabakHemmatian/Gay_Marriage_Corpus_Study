"""Compare topic distributions when using "one-hot" topic contribution
calculations, as opposed to incorporating the entire probability distribution
of term-related topics into the topic contribution calculation.

https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/issues/19
"""
from collections import defaultdict
import gensim
import math
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
# thoth-python.org
import thoth.thoth as thoth
from config import *
assert ENTIRE_CORPUS
from ModelEstimation import LDAModel
from Utils import *

def get_topic_contribution_estimations():
    ldam=LDAModel()
    ldam.Define_Sets()
    ldam.dictionary=gensim.corpora.Dictionary.load(path+"/RC_LDA_50_True.lda.id2word")
    ldam.get_model()
    ldam.relevant_year, ldam.cumm_rel_year=Yearly_Counts()
    
    for one_hot in (True, False):
        ldam.one_hot=one_hot
        for i in range(10):
            ldam.fns["topic_cont"]="{}/yr_topic_cont_{}_{}".format(ldam.output_path,
                 "one-hot" if one_hot else "distributions", i)
            ldam.Get_Topic_Contribution()

def get_distances(distance_metric="jsd"):
    distances=np.zeros((20,20))
    dist_files=[ "{}/yr_topic_cont_distributions_{}".format(output_path, i) for
                 i in range(10) ]
    onehot_files=[ "{}/yr_topic_cont_one-hot_{}".format(output_path, i) for i in
                   range(10) ]
    files=dict(zip(range(20), dist_files+onehot_files)) 
    for i in range(20):
        for j in range(20):
            dist1=np.loadtxt(files[i])
            dist2=np.loadtxt(files[j])
            contributions=[]
            for dist in (dist1, dist2):
                contributions.append([ sum(dist[:,i_]) for i_ in
                                       range(num_topics) ])
            aprob=thoth.prob_from_array(contributions[0])
            bprob=thoth.prob_from_array(contributions[1])
            if distance_metric=="jsd":
                distance=thoth.jsd(aprob, bprob, .5)
            if abs(distance)<=1e-14: # Arbitrary cut-off to deal with floating
            # point precision errors around 0
                distance=0
            distances[i][j]=distance
            distances[j][i]=distance
    with open("distances", "wb") as wfh:
        pickle.dump(distances, wfh)
    return distances

def compare_topic_distributions(distances):
    log_=lambda x:math.log(x) if x!=0 else 0

    # Get distances for within "distribution-mode" runs, within one-hot runs,
    # and between runs
    # Within "distribution-mode" runs
    within_dist=defaultdict(float)
    for i in range(10):
        for j in range(10):
            if i==j or str(sorted((i, j))) in within_dist:
                continue
            within_dist[str(sorted((i, j)))]=distances[i][j]
    within_dist=within_dist.values()
    min_, max_, mean, std=min(within_dist), max(within_dist), np.mean(within_dist), np.std(within_dist)
    print """Within "distribution mode":

    Mean: {}
    SD: {}
    Min: {}
    Max: {}
    """.format(mean, std, min_, max_)
    within_dist=map(log_, within_dist)

    # Within one-hot runs
    within_onehot=defaultdict(float)
    for i in range(10, 20):
        for j in range(10, 20):
            if i==j or str(sorted((i, j))) in within_onehot:
                continue
            within_onehot[str(sorted((i, j)))]=distances[i][j]
    within_onehot=within_onehot.values()
    min_, max_, mean, std=min(within_onehot), max(within_onehot), np.mean(within_onehot), np.std(within_onehot)
    print """Within one-hot mode:

    Mean: {}
    SD: {}
    Min: {}
    Max: {}
    """.format(mean, std, min_, max_)
    within_onehot=map(log_, within_onehot)

    # Between runs
    between=[]
    for i in range(10):
        for j in range(10, 20):
            between.append(distances[i][j])
    min_, max_, mean, std=min(between), max(between), np.mean(between), np.std(between)
    print """Between modes:

    Mean: {}
    SD: {}
    Min: {}
    Max: {}
    """.format(mean, std, min_, max_)
    between=map(log_, between)

    # Plot
    fig=plt.figure()
    ax=fig.add_subplot(111)
    min_=min(within_dist+within_onehot+between)-1
    max_=max(within_dist+within_onehot+between)+1
    ax.set_xlim(min_, max_)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    plt.hlines(.5, min_, max_)

    plt.plot(within_dist, [.5]*len(within_dist), "go", ms=5, alpha=.8,
             label="Within distribution mode")
    plt.plot(within_onehot, [.5]*len(within_onehot), "bo", ms=5, alpha=.8,
             label="Within one-hot mode")
    plt.plot(between, [.5]*len(between), "ro", ms=5, alpha=.8,
             label="Between modes")
    plt.legend()
    plt.title("Log(JSD) between different runs of the topic contribution calculation")
    plt.show()

def compare_top_topics():
    ldam=LDAModel()
    top_topic_no=int(ceil(ldam.sample_topics*ldam.num_topics))
    top_topics=np.empty((20, top_topic_no))
    dist_files=[ "{}/yr_topic_cont_distributions_{}".format(output_path, i) for
                 i in range(10) ]
    onehot_files=[ "{}/yr_topic_cont_one-hot_{}".format(output_path, i) for i in
                   range(10) ]
    files=dict(zip(range(20), dist_files+onehot_files)) 
    for i in range(20):
        yr_topic_cont=np.loadtxt(files[i])
        ldam.get_top_topics(yr_topic_cont=yr_topic_cont)
        top_topics[i]=ldam.top_topics
    return top_topics 

if __name__=="__main__":
    distances=get_distances()
    compare_topic_distributions(distances)
