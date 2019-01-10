from __future__ import print_function
import numpy
import pickle
import sys
import ast
import csv
from config import num_topics,num_pop
import matplotlib.pyplot as plt

## load impactful comments' topic assignments

# create a dictionary for counting topics representative of popular comments
pop_dict = {key: 0 for key in range(0,number_of_topics)}
scoring = []

with open("popular_comments_proc.csv",'r+b') as csvfile:
    reader = csv.reader(csvfile)
    pop_comm_topic = [] # initialize list for topic contributions
    text_comm_topic = [] # initialize list for comment texts
    # read human data for sampled comments one by one
    for idx,row in enumerate(reader):
        # ignore headers
        if idx != 0:
            tuples = ast.literal_eval(row[4])
            pop_comm_topic.append(tuples)
            text_comm_topic.append(row[0])
            scoring.append(row[3])

## Sample popular comments most associated with each topic for expert ratings

# how many comments from a topic have been assigned to a rater so far
topic_counter = {key: 0 for key in range(0,number_of_topics)}
index_pr_topic = {key: [] for key in range(0,number_of_topics)} # indices for comments from a topic
# per rater topic counter
top_cr_per_rater = {key: {keys: 0 for keys in range(0,number_of_topics)} for key in range(8)}
# which indices have been assigned to someone
seen = []
# which indices have been assigned to a particular rater
seen_by_rater = {key: [] for key in range(8)}
# which indices have been included in the overlap between raters
overlap = []
# how many comments are there in our dataset for a certain topic
pop_exp_dicts = {key: 0 for key in range(0,number_of_topics)}
sums = 0 # make sure we're counting properly
for ind,comment in enumerate(pop_comm_topic):
    pop_exp_dicts[comment[0][0]] += 1

for topic in pop_exp_dicts.iterkeys():
    sums+= pop_exp_dicts[topic]

# sanity checks
print(sums)
print(max(pop_exp_dicts.values()))
# topic with 1 comment, topic with 310. Welcome to our lives!

for rater in range(8): # for each rater
    rater_counter = 0 # reset how many they've seen
    topic_to_seek = 0 # reset which topic we're looking at
    # no overlap first
    with open("exp_pop_comm_"+str(rater)+".csv",'w+b') as csvwrite:
        writer = csv.writer(csvwrite)
        while rater_counter < 100: # if 100 comments haven't been assinged to this person
            # rows will be index, text, topic, contrib
            while topic_counter[topic_to_seek % number_of_topics] == pop_exp_dicts[topic_to_seek % number_of_topics]: # see if more comments from this topic exist
                topic_to_seek += 1 # if not, change the topic
            for idx,comment in enumerate(pop_comm_topic):
                if pop_comm_topic[idx][0][0] == topic_to_seek % number_of_topics and idx not in seen:
                    writer.writerow([idx,text_comm_topic[idx],pop_comm_topic[idx][0][0],pop_comm_topic[idx][0][1]])
                    rater_counter += 1 # update the number of comments for this rater
                    index_pr_topic[topic_to_seek % number_of_topics].append(idx)
                    topic_counter[topic_to_seek % number_of_topics] += 1 # update counts for the relevant topic
                    top_cr_per_rater[rater][topic_to_seek % number_of_topics] += 1
                    topic_to_seek += 1 # update topic to seek counter
                    seen.append(idx) # add comment to the list of assigned comments
                    seen_by_rater[rater].append(idx) # for the user as well
                    break # does this break out of for? I hope so

overlap_topic_cr = {key: 0 for key in range(0,number_of_topics)}

# see distribution of comments per topic
print("All impactful comments per assigned top topic")
print(topic_counter)
plt.bar(topic_counter.keys(), topic_counter.values(), color='g')
plt.show()

## Sample impactful comments evenly from different topics for raters with 20% overlap

topic_to_seek = 0 # reset which topic we're looking at from the previous loop
for rater in range(8): # for each rater
    overlap_counter = 0 # counter for overlap indices for a particular rater

    # now overlap
    with open("exp_pop_comm_"+str(rater)+".csv",'a+b') as csvwrite:
        writer = csv.writer(csvwrite)
        while overlap_counter < 20: # if 20 overlap comments weren't sampled

            # sample one of the comments from the same topic
            options = [x for x in index_pr_topic[topic_to_seek % number_of_topics] if x not in seen_by_rater[rater] and x not in overlap]
            if len(options) > 0:
                proposed = numpy.random.choice(options)
                writer.writerow([proposed,text_comm_topic[proposed],pop_comm_topic[proposed][0][0],pop_comm_topic[proposed][0][1]])
                overlap_counter += 1 # update counter
                overlap.append(proposed) # record the sampled comment
                overlap_topic_cr[topic_to_seek % number_of_topics] += 1 # record the number of comments in overlap from that topic
                topic_to_seek += 1 # update topic you're looking for

            else:
                topic_to_seek += 1

# see distribution of comments to be rated per topic
print("Subsampled impactful comments per assigned top topic")
print(overlap_topic_cr)
plt.bar(overlap_topic_cr.keys(), overlap_topic_cr.values(), color='g')
plt.show()

# NOTE: this includes topics for ALL [num_pop] comments

#comp for exclusion by questions, time for exclusion by reading the instructions
dfull = pickle.load(open("mturk_data-expert.pkl","rb"))
dfull = dfull.loc[dfull['category'] >= 0] # exclude "neither" comments (default)

# NOTE: If we wish to include "neither" comments from training and test
# dfull = dfull.loc[dfull['category'] >= 0]

## Print statistics for categorization of comments by raters
full_count = {-1:0,0:0,1:0}
means = []
for idx,comment in enumerate(dfull.index):
    full_count[dfull.category[comment]] += 1

print("Mean and standard deviation of rated impactful comments (respectively):")
print(numpy.mean(dfull["mean"].values))
print(numpy.std(dfull["mean"].values))

print("Their categorical distribution:")
print(full_count)

## Distribution across topics of rated comments for PREPROCESSED comments

# which indices are in the sets? (exp_pop_comm_[RATER].csv files must be in the
# same directory)
all_indices = []
for rater in range(8):
    with open("exp_pop_comm_"+str(rater)+".csv","r+b") as csvread:
        reader = csv.reader(csvread)
        for idx,comment in enumerate(reader):
            if idx < 100 and int(comment[0]) not in all_indices:
                all_indices.append(int(comment[0]))

# Top topics among [num_pop] most impactful comments
top_topic_contrib = [] # initialize list for contributions of top topics
topics_in_general_sample = {key:0 for key in range(num_topics)}
for comment in range(num_pop):
    topics_in_general_sample[pop_comm_topic[comment][0][0]]+= 1

# Top topics among comments sampled for human ratings
topics_in_processed = {key:0 for key in range(num_topics)}
for comment in all_indices:
    topics_in_processed[pop_comm_topic[comment][0][0]]+= 1
    top_topic_contrib.append(pop_comm_topic[comment][0][1])

# Print the distributions
print("Sample of impactful comments")
print(topics_in_general_sample)
print("Rated subsample")
print(topics_in_processed)

# Plot the distributions
plt.bar(topics_in_general_sample.keys(), topics_in_general_sample.values(), color='g')
plt.xlabel("Topic ID")
plt.ylabel("Number of Sampled Comments")
plt.show()
plt.bar(topics_in_processed.keys(), topics_in_processed.values(), color='g')
plt.xlabel("Topic ID")
plt.ylabel("Number of Sampled Comments")
plt.show()

# Print mean top topic contribution
print("mean top topic contribution: "+str(numpy.mean(top_topic_contrib)))
