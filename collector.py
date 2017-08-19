import logging
import json
import os
import requests
import time
from utils import *

BASE_URL="https://graph.facebook.com/"

# A wrapper for functions to interact directly with the Graph API
class request():
    def __init__(self):
        startLog()
    
    # Returns a json object with the response to a query whose arguments
    # are specified as arguments to the function.
    def get_data(self,url=None,node=None,edge=None,**kwargs):
        assert bool(url) != bool(node)
        if url:
            assert not edge
        else:
            url=self._format_request(node,edge,**kwargs)
        logging.info('Sending request to '+url)
        response=requests.get(url)
        assert response.status_code==200
        return json.loads(response.content)

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
        nextval=json_obj['paging']['next']
        if page:
            return self.get_data(url=nextval)
        return nextval
