from lda_defaults import *
from lda_config import *

# ENTIRE_CORPUS should be True if NN is True
if NN:
    try:
        assert ENTIRE_CORPUS
    except AssertionError:
        raw_input("NN is True. Overriding config settings and setting ENTIRE_CORPUS to True. Press Enter to continue.")


