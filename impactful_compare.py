from __future__ import print_function
import csv
import sys
import os
import ast
import matplotlib.pyplot as plt

### Compare distrbutions of assigned topics among rated impactful comments
# for different [num_topics]
# Requires exp_pop_comm_[rater]_[num_topics] files (rename set filenames if need be)
# Requires popular_comments_[num_topics] file in the same folder

num_topics = 25
# the online indices
indices = [[] for x in range(8)]
for rater in range(8):
    with open(os.path.abspath("exp_pop_comm_"+str(rater)+"_50.csv"),"r+b") as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            indices[rater].append(line[0])

topic_counter = {key:0 for key in range(num_topics)}
total_counter = {key:0 for key in range(num_topics)}

# total numbers in the comparison num_topics
with open(os.path.abspath("popular_comments_"+str(num_topics)+".csv"),"r+b") as reference:
    reader = csv.reader(reference)
    for idx,line in enumerate(reader):
        if idx != 0:
            tuples = ast.literal_eval(line[4])
            if len(tuples) != 0:
                total_counter[tuples[0][0]] += 1

print(total_counter)
plt.bar(total_counter.keys(), total_counter.values(), color='g')
plt.title('Total Number of Comments Per Topic Among Most up(down)-voted Comments')
plt.xlabel('Topic')
plt.ylabel('Number of Comments')

plt.show()

# the comparison indices
for rater in indices:
    for comment in rater:
        with open(os.path.abspath("popular_comments_"+str(num_topics)+".csv"),"r+b") as reference:
            reader = csv.reader(reference)
            for idx,line in enumerate(reader):
                if idx != 0 and idx == int(comment):
                    tuples = ast.literal_eval(line[4])
                    topic_counter[tuples[0][0]] += 1

print(topic_counter)

# plot the set of sampled topics in the rated subsample for comparison num_topics
plt.bar(topic_counter.keys(), topic_counter.values(), color='g')
plt.title('Comments Sampled for Rating Per Topic Among Most up(down)-voted Comments')
plt.xlabel('Topic')
plt.ylabel('Number of Comments')
plt.show()
