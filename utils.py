import random
import re
import string
import time

WORKING_DIR='./'
ACCESS_TOKEN_FILE='/Users/sabinasloman/Desktop/.access_token'

def getRelPath(fn):
    return WORKING_DIR+fn

def getAccessToken():
    with open(ACCESS_TOKEN_FILE,'r') as fh:
        return fh.read().strip()

def to_ascii(s):
    return s.encode('ascii','ignore')

def zip_(fn,dir_=False):
    import os

    rep_dict=dict(fn=fn)
    if dir_:
        if len(os.listdir(dir_))>0:
            os.system('tar -cf %(fn)s.tar %(fn)s/* && bzip2 %(fn)s.tar && rm -rf %(fn)s'%rep_dict)
        else:
            os.system('rm -r %(fn)s'%rep_dict)
    else:
        os.system('bzip2 -vf %(fn)s'%rep_dict)

def unzip_(fn,dir_=False):
    import os

    rep_dict=dict(fn=fn)
    if dir_:
        os.system('bunzip2 %(fn)s.tar.bz2 && tar -xf %(fn)s.tar && rm %(fn)s.tar'%rep_dict)
    else:
       os.system('bunzip2 -v %(fn)s.bz2'%rep_dict)

def startLog():
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def iso8601_to_unix(iso8601_str):
    return time.mktime(time.strptime(iso8601_str,'%Y-%m-%dT%H:%M:%S'))

def unix_to_iso8601(unix_str):
    return time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(unix_str))

def getFilterRegex():
    return re.compile("(gay|homosexual|homophile|fag|faggot|fagot|queer|homo|fairy|nance|pansy|queen).*(marry|marri)", re.IGNORECASE)
