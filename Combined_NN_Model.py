#!/usr/bin/python27
# -*- coding: utf-8 -*-

### import the required modules and functions

from __future__ import print_function
from config import *
from reddit_parser import Parser
from ModelEstimation import NNModel

### set default file encoding

reload(sys)
sys.setdefaultencoding('utf-8')

### determine kind of network  ###

classifier = False # If False, the neural network will model language. If true, it will perform classification on comments
pretrained = False # whether there is language model pre-training. Can only be set to True if classifier is also True

# NOTE: For classifier pretraining, the code should first be run with classifier = False & pretrained = False and param_path should be set according to the output_path that results from the first run of the code
# NOTE: if pre-training is on, network hyperparameters should not be changed from the ones previously used for language modeling

### Neural Network hyperparameters
epochs = 3 # number of epochs
learning_rate = 0.003 # learning rate
batchSz = 50 # number of parallel batches
embedSz = 128 # embedding size
hiddenSz = 512 # number of units in the recurrent layer
ff1Sz = 1000 # number of units in the first feedforward layer
ff2Sz = 1000 # number of units in the second feedforward layer
keepP = 0.5 # 1 - dropout rate
early_stopping = True # whether to stop training if development set perplexity is going up
l2regularization = False # whether the model will be penalized for longer weight vectors. Helps prevent overfitting
alpha = 0.01 # L2 regularization constant

### Paths

## where the data is

# NOTE: if not available, download from http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable accordingly

## where the output will be stored

# NOTE: To avoid confusion between different kinds of models, always include cl and pre in the output directory's name. After those, record the variables most important to your iteration

output_path = path+"/"+"cl_"+str(classifier)+"_pre_"+str(pretrained)+"_e_"+str(epochs)+"_"+"hd"+"_"+str(hiddenSz)
if not os.path.exists(output_path):
    print("Creating directory to store the output")
    os.makedirs(output_path)

## where the saved parameters are

# NOTE: Enter manually. Only matters if classifier = True and pretrained = True

param_path = path+"/cl_False_pre_False_e_3_hd_512/"
if pretrained == True:
    if not os.path.exists(param_path):
        raise Exception("Could not find the saved parameters.")

## create file to record the performance

perf = open(output_path+"/Performance",'a+')

# record the kind of network

print("***",file=perf)
print("classifier = " + str(classifier),file=perf)
print("pretrained = " + str(pretrained),file=perf)

### Preprocessing ###

### determine the set of stopwords used in preprocessing

stop = set(nltk.corpus.stopwords.words('english'))

### call the parsing function

## NOTE: If NN = False, will pre-process data for LDA

theparser=Parser()
theparser.Parse_Rel_RC_Comments()

### call the function for calculating the percentage of relevant comments
theparser.Perc_Rel_RC_Comment()

### create training, development and test sets

# NOTE: Always index training set first.
# NOTE: For valid analysis results, maximum vocabulary size and frequency filter should not be changed in between the creation of various sets

training_fraction = 0.80 # fraction of the data that is used for training

## Determine the comments that will comprise various sets

# NOTE: [1 - training_fraction] fraction of the dataset will be divided randomly and equally into evaluation and test sets

nnmodel=NNModel(training_fraction=training_fraction)
nnmodel.Define_Sets()

## Read and index the content of comments in each set

print("Vocabulary size = " + str(MaxVocab),file=perf)
print("Frequency filter = below " + str(FrequencyFilter),file=perf)

for set_key in nnmodel.set_key_list:
    nnmodel.Index_Set(set_key)

## if classifying, load comment labels from file
if classifier == True:
    vote['train'],vote['dev'],vote['test'] = Get_Votes(path)

### Language Modeling and Classification Neural Network ###

### create the computation graph

## check key hyperparameters for the correct type

assert 0 < learning_rate and 1 > learning_rate
assert type(batchSz) is int
assert type(embedSz) is int
assert type(hiddenSz) is int
assert type(ff1Sz) is int
assert type(ff2Sz) is int
assert 0 < keepP and 1 >= keepP
assert type(l2regularization) is bool
if l2regularization == True:
    assert 0 < alpha and 1 > alpha
assert type(early_stopping) is bool

## record the hyperparameters

print("Learning_rate = " + str(learning_rate),file=perf)
print("Batch size = " + str(batchSz),file=perf)
print("Embedding size = " + str(embedSz),file=perf)
print("Recurrent layer size = " + str(hiddenSz),file=perf)
print("1st feedforward layer size = " + str(ff1Sz),file=perf)
print("2nd feedforward layer size = " + str(ff2Sz),file=perf)
print("Dropout rate = " + str(1 - keepP),file=perf)
print("L2 regularization = " + str(l2regularization),file=perf)
print("L2 regularization constant = " + str(alpha),file=perf)
print("Early stopping = " + str(early_stopping),file=perf)

## create placeholders for input, output, loss weights and dropout rate

inpt = tf.placeholder(tf.int32, shape=[None,None])
answr = tf.placeholder(tf.int32, shape=[None,None])
loss_weight = tf.placeholder(tf.float32, shape=[None,None])
DOutRate = tf.placeholder(tf.float32)

## set up the graph parameters

# for pre-trained classification network, load parameters from file

if classifier == True and pretrained == True:
    print("Loading parameter estimates from file")
    embeddings = np.loadtxt(param_path+"embeddings",dtype='float32')
    pre_state = np.loadtxt(param_path+"state",dtype='float32')
    weights1 = np.loadtxt(param_path+"weights1",dtype='float32')
    biases1 = np.loadtxt(param_path+"biases1",dtype='float32')
    weights2 = np.loadtxt(param_path+"weights2",dtype='float32')
    biases2 = np.loadtxt(param_path+"biases2",dtype='float32')
    weights3 = np.loadtxt(param_path+"weights3",dtype='float32')
    biases3 = np.loadtxt(param_path+"biases3",dtype='float32')
else:
    print("Initializing parameter estimates")

# initial embeddings

if classifier == True and pretrained == True:
    E = tf.Variable(embeddings)
else:
    E = tf.Variable(tf.random_normal([len(V), embedSz], stddev = 0.1))

# look up the embeddings
embed = tf.nn.embedding_lookup(E, inpt)

# calculate sum of the weights for l2regularization
if l2regularization == True:
    sum_weights = tf.nn.l2_loss(embed)

# define the recurrent layer (Gated Recurrent Unit)
rnn= tf.contrib.rnn.GRUCell(hiddenSz)

if classifier == True and pretrained == True:
    initialState = pre_state # load pretrained state
else:
    initialState = rnn.zero_state(batchSz, tf.float32)

output, nextState = tf.nn.dynamic_rnn(rnn, embed,initial_state=initialState)

# update sum of the weights for l2regularization
if l2regularization == True:
    sum_weights = sum_weights + tf.nn.l2_loss(nextState)

if classifier == False: # language modeling
    # create weights and biases for three feedforward layers
    W1 = tf.Variable(tf.random_normal([hiddenSz,ff1Sz], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([ff1Sz], stddev=0.1))
    l1logits = tf.nn.relu(tf.tensordot(output,W1,[[2],[0]])+b1)
    l1Output = tf.nn.dropout(l1logits,DOutRate) # apply dropout
    W2 = tf.Variable(tf.random_normal([ff1Sz,ff2Sz], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([ff2Sz], stddev=0.1))
    l2Output = tf.nn.relu(tf.tensordot(l1Output,W2,[[2],[0]])+b2)
    W3 = tf.Variable(tf.random_normal([ff2Sz,len(V)], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([len(V)], stddev=0.1))

    # update parameter vector lengths for l2regularization
    if l2regularization == True:
        for vector in [W1,b1,W2,b2,W3,b3]:
            sum_weights = sum_weights + tf.nn.l2_loss(vector)

    ## calculate loss

    # calculate logits
    logits = tf.tensordot(l2Output,W3,[[2],[0]])+b3

    # calculate sequence cross-entropy loss
    xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=answr,weights=loss_weight)

    if l2regularization == True:
        loss = tf.reduce_mean(xEnt) + (alpha * sum_weights)
    else:
        loss = tf.reduce_mean(xEnt)

elif classifier == True: # classification
    if pretrained == False: # if initializing parameters
        # create weights and biases for three feedforward layers
        W1 = tf.Variable(tf.random_normal([hiddenSz,ff1Sz], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([ff1Sz], stddev=0.1))
        l1logits = tf.nn.relu(tf.matmul(nextState,W1)+b1)
        l1Output = tf.nn.dropout(l1logits,keepP) # apply dropout
        W2 = tf.Variable(tf.random_normal([ff1Sz,ff2Sz], stddev=0.1))
        b2 = tf.Variable(tf.random_normal([ff2Sz], stddev=0.1))
        l2Output = tf.nn.relu(tf.matmul(l1Output,W2)+b2)
        W3 = tf.Variable(tf.random_normal([ff2Sz,len(V)], stddev=0.1))
        b3 = tf.Variable(tf.random_normal([len(V)], stddev=0.1))
    if pretrained == True: # if using pre-trained weights
        W1 = tf.Variable(weights1)
        b1 = tf.Variable(biases1)
        l1logits = tf.nn.relu(tf.matmul(nextState,W1)+b1)
        l1Output = tf.nn.dropout(l1logits,keepP) # apply dropout
        W2 = tf.Variable(weights2)
        b2 = tf.Variable(biases2)
        l2Output = tf.nn.relu(tf.matmul(l1Output,W2)+b2)
        W3 = tf.Variable(weights3)
        b3 = tf.Variable(biases3)

    l3Output = tf.nn.relu(tf.matmul(l2Output,W3)+b3)

    # create weights and biases for a vote prediction output layer
    W4 = tf.Variable(tf.random_normal([len(V),3], stddev=0.1))
    b4 = tf.Variable(tf.random_normal([3],stddev=0.1))

    # update parameter vector lengths for l2regularization
    if l2regularization == True:
        for vector in [W1,b1,W2,b2,W3,b3,W4,b4]:
            sum_weights = sum_weights + tf.nn.l2_loss(vector)

    ### calculate loss

    # calculate logits
    logits = tf.matmul(l3Output,W4)+b4

    # softmax
    prbs = tf.nn.softmax(logits)

    # calculate cross-entropy loss
    xEnt = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=answr)

    if l2regularization == True:
        loss = tf.reduce_mean(xEnt) + (alpha * sum_weights)
    else:
        loss = tf.reduce_mean(xEnt)

    # calculate accuracy
    numCorrect = tf.equal(tf.argmax(prbs,1), tf.argmax(answr,1))
    numCorrect = tf.reduce_sum(tf.cast(numCorrect, tf.float32))

## training with AdamOptimizer

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

## create the session and initialize the variables

config = tf.ConfigProto(device_count = {'GPU': 0}) # Use only CPU (due to overly large matrices)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
if pretrained == False:
    state = sess.run(initialState)

### Train and Test the Neural Network ###

## create a list of comment lengths

for set_key in set_key_list:
    for i,x in enumerate(indexes[set_key]):
        lengths[set_key].append(len(indexes[set_key][i]))
    Max[set_key] = max(lengths[set_key]) # maximum length of a comment in this set
Max_l = max(Max['train'],Max['dev'],Max['test']) # max length of a comment in the whole dataset

## initialize vectors to store set accuracies or perplexities

if classifier == True:
    accuracy = {key: [] for key in set_key_list}
    for set_key in set_key_list:
        accuracy[set_key] = np.empty(epochs)
else:
    perplexity = {key: [] for key in set_key_list}
    for set_key in set_key_list:
        perplexity[set_key] = np.empty(epochs)

### Train and test for the determined number of epochs

print("Number of epochs: "+str(epochs))
print("Number of epochs: "+str(epochs),file=perf)

for k in range(epochs): # for each epoch

    # timer
    print("Started epoch "+str(k+1)+" at "+time.strftime('%l:%M%p'))

    for set_key in set_key_list: # for each set

        if classifier == True: # if classifying
            TotalCorr = 0 # reset number of correctly classified examples
        else: # if modeling language
            Epoch_Loss = 0 # reset the loss

        # initialize vectors for feeding data and desired output
        inputs = np.zeros([batchSz,Max_l])

        if classifier == True:
            answers = np.zeros([batchSz,3],dtype=np.int32)
        else:
            answers = np.zeros([batchSz,Max_l])
            loss_weights = np.zeros([batchSz,Max_l])

        # batch counters
        j = 0 # batch comment counter
        p = 0 # batch counter

        for i in range(len(indexes[set_key])): # for each comment in the set
            inputs[j,:lengths[set_key][i]] = indexes[set_key][i]

            if classifier == True:
                answers[j,:] = vote[set_key][i]
            else:
                answers[j,:lengths[set_key][i]-1] = indexes[set_key][i][1:]
                loss_weights[j,:lengths[set_key][i]] = 1

            j += 1 # update batch comment counter
            if j == batchSz - 1: # if the current batch is filled

                if classifier == True: # if classifying
                    if set_key == 'train':
                        # train on the examples
                        _,outputs,next,_,Corr = sess.run([train,output,nextState,loss,numCorrect],feed_dict={inpt:inputs,answr:answers,DOutRate:keepP})
                    else:
                        # test on development or test set
                        _,Corr = sess.run([loss,numCorrect],feed_dict={inpt:inputs,answr:answers,DOutRate:1})
                else: # if doing language modeling
                    if set_key == 'train':
                        # train on the examples
                        _,outputs,next,Batch_Loss = sess.run([train,output,nextState,loss],feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:keepP})
                    else:
                        # test on development or test set
                        Batch_Loss = sess.run(loss,feed_dict={inpt:inputs,answr:answers,loss_weight:loss_weights,DOutRate:1})

                j = 0 # reset batch comment counter
                p += 1 # update batch counter

                # reset the input/label containers
                inputs = np.zeros([batchSz,Max_l])
                if classifier == True:
                    answers = np.zeros([batchSz,3],dtype=np.int32)
                else:
                    answers = np.zeros([batchSz,Max_l])
                    loss_weights = np.zeros([batchSz,Max_l])

                # update the GRU state
                state = next # update the GRU state

                # update total number of correctly classified examples or total loss based on the processed batch
                if classifier == True:
                    TotalCorr += Corr
                else:
                    Epoch_Loss += Batch_Loss

            # during language modeling training, every 10000 comments or at the end of training, save the weights
            if classifier == False:
                if set_key == 'train' and ((i+1) % 10000 == 0 or i == len(indexes['train']) - 1):

                    # retrieve learned weights
                    embeddings,weights1,weights2,weights3,biases1,biases2,biases3 = sess.run([E,W1,W2,W3,b1,b2,b3])
                    embeddings = np.asarray(embeddings)
                    outputs = np.asarray(outputs)
                    weights1 = np.asarray(weights1)
                    weights2 = np.asarray(weights2)
                    weights3 = np.asarray(weights3)
                    biases1 = np.asarray(biases1)
                    biases2 = np.asarray(biases2)
                    biases3 = np.asarray(biases3)
                    # define a list of the retrieved variables
                    weights = ["embeddings","state","weights1","weights2","weights3","biases1","biases2","biases3"]
                    # write them to file
                    for variable in weights:
                        np.savetxt(output_path+"/"+variable, eval(variable))

        if classifier == True: # calculate set accuracy for the current epoch and save the value
            accuracy[set_key][k] = float(TotalCorr) / float( p * batchSz )
            print("Accuracy on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(accuracy[set_key][k]))
            print("Accuracy on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(accuracy[set_key][k]),file=perf)

        else: # calculate set perplexity for the current epoch and save the value
            # calculate set perplexity
            perplexity[set_key][k] = np.exp(Epoch_Loss / p)
            print("Perplexity on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]))
            print("Perplexity on the " + set_key + " set (Epoch " +str(k+1)+"): "+ str(perplexity[set_key][k]),file=perf)

    ## early stopping
    if early_stopping == True:
        if classifier == True:
            # if development set accuracy is decreasing, stop training to prevent overfitting
            if k != 0 and accuracy['dev'][k] < accuracy['dev'][k-1]:
                break
        else:
            # if development set perplexity is increasing, stop training to prevent overfitting
            if k != 0 and perplexity['dev'][k] > perplexity['dev'][k-1]:
                break

# timer
print("Finishing time:" + time.strftime('%l:%M%p'))
# close the performance file
perf.close()
