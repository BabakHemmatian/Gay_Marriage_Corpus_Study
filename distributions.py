"""Functions to uncover how our data is distributed across platforms,
years, etc.
"""

import json
import os
import pandas as pd
from config import *
from utils import *

"""Answers the questions:
- How many posts have we retrieved, grouped by source and year?
- How many comments have we retrieved, grouped by source and year?

Saves to disk two pandas DataFrames in the PWD:
- n_posts.pandas stores the number of posts grouped by source and year.
- n_comments.pandas stores the number of comments grouped by source and
  year.

The sources and years to report can be specified by passing them as
arguments. Otherwise, the function looks to the settings in get_data.

Example:
> distributions.by_source_and_year(sources=[ 'TheHill', 'wsj', 'usatoday' ],
  years=[ 2015, 2016 ])

To read the DataFrames:

> import pandas as pd
> pd.read_pickle('n_posts.pandas')
"""
def by_source_and_year(sources=None,years=None):
    if isinstance(sources,type(None)):
        from get_data import sources
    if isinstance(years,type(None)):
        from get_data import years

    nump,numc=dict(),dict()
    for source in sources:
        for year in years:
            for d in nump,numc:
                d[source]=d.get(source,dict())
                d[source][year]=0
    for year in years:
        for f in listdir_(DATA_DIR):
            bname=f.replace('.tar.bz2','')
            source,year_=bname.split('-')
            fname=DATA_DIR+bname
            if year_==str(year):
                unzip_(fname,dir_=True)
                for f_ in listdir_(fname):
                    nump[source][year]+=1
                    with open('{}/{}'.format(fname,f_)) as fh:
                       json_obj=json.loads(fh.read())
                    numc[source][year]+=len(json_obj)
                zip_(fname,dir_=True)

    df_np=pd.DataFrame(nump)
    df_nc=pd.DataFrame(numc)
    df_np.to_pickle('n_posts.pandas')
    df_nc.to_pickle('n_comments.pandas')
