"""Compare topic distributions when using "one-hot" topic contribution
calculations, as opposed to incorporating the entire probability distribution
of term-related topics into the topic contribution calculation.

https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/issues/19
"""
import numpy as np
# thoth-python.org
import thoth.thoth as thoth
from config import *

def compare_topic_distributions(file1, file2, distance_metric="jsd"):
    dist1, dist2=np.loadtxt(file1), np.loadtxt(file2)
    contributions=[]
    for dist in (dist1, dist2):
        contributions.append([ sum(dist[:,i]) for i in range(num_topics) ])
    aprob=thoth.prob_from_array(contributions[0])
    bprob=thoth.prob_from_array(contributions[1])
    if distance_metric=="jsd":
        return thoth.jsd(aprob, bprob, .5)

if __name__=="__main__":
    print compare_topic_distributions("{}/yr_topic_cont_one-hot".format(output_path),
                                      "{}/yr_topic_cont_distributions".format(output_path))
