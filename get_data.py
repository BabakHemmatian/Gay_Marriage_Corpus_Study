import json
import os
import collector
from utils import *

# Choose nine relatively popular political platforms that fall at different
# points along the political spectrum. Platform choice inspired by:
# Faris, Robert and Roberts, Hal and Etling, Bruce and Bourassa, Nikki and 
# Zuckerman, Ethan and Benkler, Yochai, Partisanship, Propaganda, and 
# Disinformation: Online Media and the 2016 U.S. Presidential Election 
# (August 2017). Berkman Klein Center Research Publication 2017-6. Available
# at SSRN.

# Centrist publications
media=[ 'TheHill', 'wsj', 'usatoday' ]
# Liberal-ish publications
media+=[ 'HuffPost', 'nytimes', 'washingtonpost' ]
# Conservative-ish publications
media+=[ 'realclearpolitics', 'nationalreview', 'FoxNews' ]

api_bind=collector.request()

def filter_posts_by_regex(d):
    if 'message' not in d:
        return False
    return len(getFilterRegex().findall(d['message']))>0

def get_children(cmt):
    id_=cmt['id']
    json_obj=api_bind.get_data(node=id_,edge='comments',limit=100)
    child_data=json_obj['data']
    while len(json_obj['data'])>0:
        json_obj=api_bind.get_next_val(json_obj,page=True)
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

if __name__=='__main__':
    for source in media:
        for year in range(2008,2017):
            dir_=getRelPath('{}-{}'.format(source,year))
            os.system('mkdir -p {}'.format(dir_))
            # TODO: Confusing, but seemingly harmless. Look into cases like the
            # following:
            ## $ head wsj-2012
            ## > [{"created_time": "2013-01-01T02:37:12+0000",...
            ## $ tail wsj-2013
            ## > {"created_time": "2013-01-01T17:26:49+0000",...
            json_obj=api_bind.get_data(node=source,edge='posts',
                                        since=iso8601_to_unix(
                                        '{}-01-01T00:00:00'.format(year)
                                        ),
                                        until=iso8601_to_unix(
                                        '{}-12-31T23:59:59'.format(year)
                                        ),
                                        limit=100)
            data=[]
            data+=[ d for d in json_obj['data'] if filter_posts_by_regex(d) ]
            while len(json_obj['data'])>0:
                json_obj=api_bind.get_next_val(json_obj,page=True)
                if not json_obj:
                    break
                data+=[ d for d in json_obj['data'] if filter_posts_by_regex(d) 
                    ]
            for post in data:
                id_=post['id']
                json_obj=api_bind.get_data(node=id_,edge='comments',limit=100)
                cmt_data=[]
                some_cmt_data=json_obj['data']
                # Recusively get replies to top-level comments
                get_all_children(some_cmt_data)
                cmt_data+=some_cmt_data
                while len(json_obj['data'])>0:
                    json_obj=api_bind.get_next_val(json_obj,page=True)
                    if not json_obj:
                        break
                    some_cmt_data=json_obj['data']
                    # Recursively get replies to top-level comments
                    get_all_children(some_cmt_data)
                    cmt_data+=some_cmt_data
                with open('{}/{}'.format(dir_,id_),'w') as fh:
                    json.dump(cmt_data,fh)
            zip_(dir_,dir_=True)
