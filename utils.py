import os
import random
import re
import string
import time
from config import *

def getRelPath(fn):
    return WORKING_DIR+fn

def getAccessToken():
    if not os.path.exists(ACCESS_TOKEN_FILE):
        return raw_input('Enter access token here: ').strip()
    with open(ACCESS_TOKEN_FILE,'r') as fh:
        return fh.read().strip()

def getAppID():
    if not os.path.exists(APP_ID_FILE):
        return raw_input('Enter app ID here: ').strip()
    with open(APP_ID_FILE,'r') as fh:
        return fh.read().strip()

def getAppSecret():
    if not os.path.exists(APP_SECRET_FILE):
        return raw_input('Enter app secret here: ').strip()
    with open(APP_SECRET_FILE,'r') as fh:
        return fh.read().strip()

def create_long_term_token():
    auth={'SHORT_TERM_ACCESS_TOKEN': getAccessToken(),
          'CLIENT_ID': getAppID(),
          'CLIENT_SECRET': getAppSecret()
         }
    url='https://graph.facebook.com/oauth/access_token?grant_type=fb_exchange_token&client_id=%(CLIENT_ID)s&client_secret=%(CLIENT_SECRET)s&fb_exchange_token=%(SHORT_TERM_ACCESS_TOKEN)s'%auth
    cmd='curl -X GET \"{}\"'.format(url)
    print cmd
    with os.popen(cmd) as fh:
        token=fh.read().strip()
    return token

def to_ascii(s):
    return s.encode('ascii','ignore')

def zip_(fn,dir_=False):
    import os

    rep_dict=dict(fn=fn)
    if dir_:
        if len(os.listdir(fn))>0:
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
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)
