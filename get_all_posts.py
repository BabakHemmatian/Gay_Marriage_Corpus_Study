import json
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

for source in media:
    for year in range(2008,2017):
        fn=getRelPath('{}-{}'.format(source,year))
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
        data+=json_obj['data']
        while len(json_obj['data'])>0:
            json_obj=api_bind.get_next_val(json_obj,page=True)
            data+=json_obj['data']
        with open(fn,'w') as fh:
            json.dump(data,fh)
        zip_(fn)
