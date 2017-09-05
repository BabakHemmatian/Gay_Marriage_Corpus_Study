import logging
import json
import os
import random
import requests
import string
import time
from config import *
from utils import *

# A wrapper for functions to interact directly with the Graph API
class request():
    def __init__(self, log=True, id_=None):
        if isinstance(id_,type(None)):
            id_=''.join(random.choice(string.lowercase) for _ in range(6))
        self.id_=id_
        startLog(log=log, id_=id_)
    
    # Returns a json object with the response to a query whose arguments
    # are specified as arguments to the function.
    def get_data(self,url=None,node=None,edge=None,**kwargs):
        assert bool(url) != bool(node)
        if url:
            assert not edge
        else:
            url=self._format_request(node,edge,**kwargs)
        while True:
            logging.info('Sending request to '+url)
            response=requests.get(url)
            try:
                assert response.status_code==200
                return json.loads(response.content)
            except AssertionError:
                if '#17' in response.content:
                    logging.info('Rate limit reached. Wait one hour.')
                    time.sleep(3600)
                # https://developers.facebook.com/bugs/1772936312959430/
                if '#100' in response.content:
                    logging.info('Reached absolute limit after retrieving \
                    25000 comments on this post. Stop here.')
                    return []
                logging.info('Request failed with response {}: {}. Retrying.'.format(response.status_code,response.content))
                continue

    # A helper function that formats API queries whose arguments are
    # specified as arguments to the function.
    def _format_request(self,node,edge=None,**kwargs):
        edge='' if isinstance(edge,type(None)) else edge+'/'
        header_str='{}/{}?'.format(node,edge)
        access_token_str='&access_token='+getAccessToken()
        header_str+='&'.join('{}={}'.format(k,v) for k,v in kwargs.items())
        header_str+=access_token_str
        return BASE_URL+header_str

    # Retrieve the index of the next "page" of results (used to paginate
    # through multiple pages of results)
    def get_next_val(self,json_obj,page=False):
        if 'next' not in json_obj['paging']:
            return False
        nextval=json_obj['paging']['next']
        if page:
            return self.get_data(url=nextval)
        return nextval
