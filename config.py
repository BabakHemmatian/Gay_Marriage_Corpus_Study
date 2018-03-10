from lda_defaults import *
from lda_config import *

# ENTIRE_CORPUS should be True if NN is True
if NN:
    try:
        assert ENTIRE_CORPUS
    except AssertionError:
        raw_input("NN is True. Overriding config settings and setting ENTIRE_CORPUS to True. Press Enter to continue.")
        ENTIRE_CORPUS=True

# minimum_probability should be 0 if one_hot_topic_contributions is False
if not one_hot_topic_contributions:
    try:
        assert minimum_probability==1e-8
    except AssertionError:
        raw_input("one_hot_topic_contributions is True. Overriding config settings and setting minimum_probability to 0. Press Enter to continue.")
        minimum_probability=1e-8
