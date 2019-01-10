from __future__ import print_function
import numpy as np
from math import floor
import os
from pathlib2 import Path
import csv
import sys
import time
import operator

## Number of the most popular comments sampled
num_pop = 2000
min_comm_length = 20

## Retrieve the list of cummulative monthly comment counts (same folder)
with open("RC_Count_List",'r') as f:
    timelist = []
    for line in f:
        if line.strip() != "":
            timelist.append(int(line))

## Retrieve the original text of comments (same folder)
with open("original_comm",'r') as f:
    orig_comm = []
    for line in f:
        if line.strip() != "":
            orig_comm.append(line.strip())

## Calculate the associated month and year for each comment
rel_month = np.zeros((timelist[-1],1),dtype="int32") # shape of all the relevant comments 603282,1
month_of_year = np.zeros((timelist[-1],1),dtype="int32")
rel_year = np.zeros((timelist[-1],1),dtype="int32")
# # NOTE: Only picks out comments from 2008 onwards. See comments for ways to easily change it back to including 2006-7
# print("Start Time: "+time.strftime("%Y-%m-%d %H:%M:%S"))
for rel_ind in range(timelist[-1]):
    # if rel_ind >= 756: # remove condition and unindent if including 2006-2007
    # Find the relevant month and year
    rel_month[rel_ind,0] = next((i+1 for i in range(len(timelist)) if timelist[i] > rel_ind),141)
    month_of_year[rel_ind,0] = rel_month[rel_ind,0] % 12
    if month_of_year[rel_ind,0] == 0:
        month_of_year[rel_ind,0] = 12
    rel_year[rel_ind,0] = int(2006+int(floor(rel_month[rel_ind,0]/12)))

## Sort comments based on their impact on the discourse
with open("votes",'a+') as f:
    vote_count = dict()
    abs_vote_count = dict()
    for number,line in enumerate(f):
        if line.strip() != "":
            vote_count[str(number)] = int(line)
            abs_vote_count[str(number)] = abs(int(line))
sorted_votes = sorted(abs_vote_count.items(), key=operator.itemgetter(1),reverse=True)


counter = 0
results = []
abs_results = []
popular = []
for x in sorted_votes:
    comment = orig_comm[int(x[0])]
    if len(comment.strip().split()) > min_comm_length:
        counter += 1
        results.append(vote_count[x[0]])
        abs_results.append(x[1])
        popular.append(int(x[0]))
    if counter == num_pop:
        break

# see how many have negative upvotes

counter = 0
for x in results:
    if x < 0:
        counter+=1
print("Negative upvotes count: "+str(counter))

## Pick the [num_pop] most impactful comments
top_votes = np.argsort(vote_count)[-num_pop:]
popular = orig_ind[np.argsort(vote_count)[-num_pop:]]

## Write them to file
with open("popular_comments.csv",'w+b') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text','year','month','score'])
    for number,pop_comment in enumerate(popular): # TODO: Add ups and downs to the code
        writer.writerow([orig_comm[pop_comment],rel_year[pop_comment,0],month_of_year[pop_comment,0],results[number]])
