#!/usr/local/bin/python

"""Retrieves and writes to file all comments on all relevant* Facebook posts
made by nine American news sources from 2008 through 2016.
*A post is considered relevant if its message contains a match to the regex
defined in utils.getFilterRegex().

Zipped and archived directories are created in the user's working directory
each of which contains the data and is named for each unique pair of news 
sources and years (if a news source did not make any relevant posts in a 
given year, the corresponding directory will not exist). Each file in these 
directories is named after a unique post ID, and contains JSON-formatted
data on all the comments on the corresponding post.
Example:
    $ ls $WORKING_DIR
    > wsj-2010  wsj-2011    wsj-2012    wsj-2013    wsj-2014    wsj-2015
      wsj-2016  ...
    $ ls $WORKING_DIR/wsj-2011
    > 8304333127_154962911240990    8304333127_219958831377828  ...
    $ head $WORKING_DIR/wsj-2011/8304333127_154962911240990
    > [{"created_time": "2011-06-14T13:17:32+0000", "message":...

The json library can be used to convert the contents of each file into a
single JSON object, which is essentially a list of dictionaries. Each of 
these dictionaries corresponds to a comment. The key/value pairs of the 
dictionaries contain information about the comment:

# created_time (datetime): The time this comment was made
# message (string): The comment text
# from (User; a dictionary): The person that made this comment. The
  dictionary paired with "from" itself has two key/value pairs:
  # name (string): The person's full name
  # id (numeric string): The id of this person's user account
# id (string): The comment ID
# children (Comments; a list of dictionaries): A list of the Comment objects
  that are replies to this comment. N.B.: This is the only "developer-
  generated" key/value pair, in the sense that the value is not returned by 
  the same query that returns its parent comment. Note that this key/value 
  pair does not exist in  Comment objects to which there are no direct 
  replies.

https://developers.facebook.com/docs/graph-api/reference
"""

import json
import os
import requests
from wordplay import collectors
from config import *
from utils import *

from get_sources import left, middle, right
sources=left+middle+right

# Retrieve posts from 2008 through 2016.
years=[ 2008,2009,2010,2011,2012,2013,2014,2015,2016 ]

api_bind=collectors.FacebookCollector()

def get_json(response):
    if not hasattr(response, 'content'):
        return []
    return json.loads(response.content)

def filter_posts_by_regex(d):
    if 'message' not in d:
        return False
    return len(getFilterRegex().findall(d['message']))>0

def get_children(cmt):
    id_=cmt['id']
    response=api_bind.get_data(node=id_,edge='comments',limit=100)
    json_obj=get_json(response)
    child_data=json_obj['data']
    while len(json_obj['data'])>0:
        response=api_bind.get_next_val(json_obj,page=True)
        json_obj=get_json(response)
        if not json_obj:
            break
        child_data+=json_obj['data']
    return child_data

def get_all_children(some_cmt_data):
    for cmt in some_cmt_data:
        children=get_children(cmt)
        if len(children)>0:
            cmt['children']=children
        while len(children)>0:
            grandchildren=[]
            for cmt_ in children:
                children_=get_children(cmt_)
                if len(children_)>0:
                    cmt_['children']=children_
                grandchildren+=children_
            children=grandchildren

def main():
    for source in sources:
        for year in years:
            dir_='{}{}-{}'.format(DATA_DIR,source,year)
            os.system('mkdir -p {}'.format(dir_))
            # TODO: Confusing, but seemingly harmless. Look into cases like 
            # the following:
            ## $ head wsj-2012
            ## > [{"created_time": "2013-01-01T02:37:12+0000",...
            ## $ tail wsj-2013
            ## > {"created_time": "2013-01-01T17:26:49+0000",...
            response=api_bind.get_data(node=source,edge='posts',
                                       since=iso8601_to_unix(
                                       '{}-01-01T00:00:00'.format(year)
                                       ),
                                       until=iso8601_to_unix(
                                       '{}-12-31T23:59:59'.format(year)
                                       ),
                                       limit=100)
            json_obj=get_json(response)
            data=[]
            data+=[ d for d in json_obj['data'] if filter_posts_by_regex(d) ]
            while len(json_obj['data'])>0:
                response=api_bind.get_next_val(json_obj,page=True)
                json_obj=get_json(response)
                if not json_obj:
                    break
                data+=[ d for d in json_obj['data'] if filter_posts_by_regex(d) 
                    ]
            for post in data:
                id_=post['id']
                response=api_bind.get_data(node=id_,edge='comments',limit=100)
                json_obj=get_json(response)
                cmt_data=[]
                some_cmt_data=json_obj['data']
                # Recusively get replies to top-level comments
                get_all_children(some_cmt_data)
                cmt_data+=some_cmt_data
                while len(json_obj['data'])>0:
                    response=api_bind.get_next_val(json_obj,page=True)
                    json_obj=get_json(response)
                    if not json_obj:
                        break
                    some_cmt_data=json_obj['data']
                    # Recursively get replies to top-level comments
                    get_all_children(some_cmt_data)
                    cmt_data+=some_cmt_data
                with open('{}/{}'.format(dir_,id_),'w') as fh:
                    json.dump(cmt_data,fh)
            zip_(dir_,dir_=True)
