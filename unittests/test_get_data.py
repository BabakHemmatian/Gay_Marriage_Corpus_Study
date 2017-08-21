import json
import os
import collector
import get_data
from utils import *
sample_data_dir=getRelPath('unittests/sample_data/')

def test_get_all_children():
    def _get_data(node,**kwargs):
        if os.path.exists(sample_data_dir+node):
            with open(sample_data_dir+node,'r') as f:
                return json.loads(f.read().replace('\n',''))
        else:
            return {"data":[]}
    
    with open(sample_data_dir+'some_cmt_data') as f:
       some_cmt_data=json.loads(f.read().replace('\n',''))['data']
    get_data.api_bind=collector.request()
    get_data.api_bind.get_data=_get_data
    get_data.get_all_children(some_cmt_data)
    assert 'children' in some_cmt_data[0]
    assert 'children' not in some_cmt_data[1]
    assert 'children' in some_cmt_data[2]
    assert 'children' in some_cmt_data[0]['children'][0]
    assert 'children' not in some_cmt_data[2]['children']
    assert some_cmt_data[0]['children'][0]['id']=='4567'
