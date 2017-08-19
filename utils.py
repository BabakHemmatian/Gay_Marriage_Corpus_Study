import random
import string
import time

WORKING_DIR='./'
ACCESS_TOKEN_FILE='/Users/sabinasloman/Desktop/access_token'

def getRelPath(fn):
    return WORKING_DIR+fn

def getAccessToken():
    with open(ACCESS_TOKEN_FILE,'r') as fh:
        return fh.read().strip()

def to_ascii(s):
    return s.encode('ascii','ignore')

def zip_(fn):
    import os

    os.system('bzip2 -vf {}'.format(fn))

def unzip_(fn):
    import os

    os.system('bunzip2 -v {}.bz2'.format(fn))

def startLog():
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def iso8601_to_unix(iso8601_str):
    return time.mktime(time.strptime(iso8601_str,'%Y-%m-%dT%H:%M:%S'))

def unix_to_iso8601(unix_str):
    return time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(unix_str))
