# Are we using a random subset of comments, or the whole dataset?
ENTIRE_CORPUS=False

### determine hyperparameters ###

### Pre-processing hyperparameters
MaxVocab = 2000000 # maximum size of the vocabulary
FrequencyFilter = 1 # tokens with a frequency equal or less than this number will be filtered out of the corpus
no_below = 5 # tokens that appear in less than this number of documents in corpus will be filtered out
no_above = 0.99 # tokens that appear in more than this fraction of documents in corpus will be filtered out
training_fraction = 0.99 # what percentage of data will be used for training. The rest of the dataset will be used as an evaluation set for calculating perplexity

### LDA hyperparameters
iterations = 1000 # number of times LDA posterior distributions will be sampled
num_threads = 5 # number of threads used for parallelized processing of comments. Only matters if using _Threaded functions
num_topics = 50 # number of topics to be generated in each LDA sampling
sample_topics = 0.2 # percentage of topics that will be selected for reporting based on average yearly contribution
topn = 40 # the number of high-probability words for each topic that will be exported
sample_comments = 100 # number of comments that will be sampled from top topics
min_comm_length = 20 # the minimum acceptable number of words in a sampled comment. Set to None for no length filtering
alpha = 0.1 # determines how many high probability topics will be assigned to a document in general (not to be confused with NN l2regularization constant)
minimum_probability = 0.01 # minimum acceptable probability for an output topic across corpus
eta = 0.1 # determines how many high probability words will be assigned to a topic in general
minimum_phi_value = 0.01 # determines the lower bound on per-term topic probability. Only matters if per_word_topics = True.
n_random_comments = 1500 # number of comments to sample from each year for
# training

### Paths

## where the data is
# NOTE: if not fully available on file, set Download for Parser function to True (source: http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable accordingly
file_path = os.path.abspath(sys.argv[0])
path = os.path.dirname(file_path)

## Year/month combinations to get Reddit data for
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2006,2018) # year range
for year in years:
    for month in months:
        if year==2017 and month==10: # till Sep 2017
            break

## where the output will be stored
# NOTE: To avoid confusion between different kinds of models, record the variables most important to your iteration in the folder name
output_path = path + "/LDA_Full_"+str(num_topics)
