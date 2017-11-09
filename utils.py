import os
import random
import re
import string
import subprocess
import time
from config import *

# Quirky OSX
def listdir_(dir_):
    return [ f for f in os.listdir(dir_) if f!='.DS_Store' ]

def getRelPath(fn):
    return WORKING_DIR+fn

def to_ascii(s):
    return s.encode('ascii','ignore')

def zip_(fn,dir_=False):
    dn,bn=os.path.dirname(fn),os.path.basename(fn)
    rep_dict=dict(fn=fn, dn=dn, bn=bn)
    if dir_:
        if len(listdir_(fn))>0:
            subprocess.call('cd %(dn)s && tar -cf %(bn)s.tar %(bn)s/* && bzip2 %(bn)s.tar && rm -rf %(bn)s'%rep_dict, shell=True)
        else:
            os.system('rm -r %(fn)s'%rep_dict)
    else:
        os.system('bzip2 -vf %(fn)s'%rep_dict)

def unzip_(fn,dir_=False):
    dn,bn=os.path.dirname(fn),os.path.basename(fn)
    rep_dict=dict(fn=fn,dn=dn,bn=bn)
    if dir_:
        os.system('bunzip2 %(fn)s.tar.bz2 && tar -xf %(fn)s.tar && rm %(fn)s.tar'%rep_dict)
        # Something about the system call sometimes dumps the unzipped file
        # into the PWD.
        if bn not in os.listdir(dn):
            os.system('mv %(bn)s %(dn)s'%rep_dict)
    else:
       os.system('bunzip2 -v %(fn)s.bz2'%rep_dict)

def iso8601_to_unix(iso8601_str):
    return time.mktime(time.strptime(iso8601_str,'%Y-%m-%dT%H:%M:%S'))

def unix_to_iso8601(unix_str):
    return time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(unix_str))

def getFilterRegex():
    return re.compile("^(?=.*gay|.*homosexual|.*homophile|.*fag|.*faggot|.*fagot|.*queer|.*homo|.*fairy|.*nance|.*pansy|.*queen|.*LGBT|.*GLBT|.*same.sex|.*lesbian|.*dike|.*dyke|.*butch|.*sodom|.*bisexual)(?=.*marry|.*marri|.*civil union).*$", re.I)
