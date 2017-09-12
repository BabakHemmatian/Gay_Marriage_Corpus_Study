# Faris, Robert and Roberts, Hal and Etling, Bruce and Bourassa, Nikki and 
# Zuckerman, Ethan and Benkler, Yochai, Partisanship, Propaganda, and 
# Disinformation: Online Media and the 2016 U.S. Presidential Election 
# (August 2017). Berkman Klein Center Research Publication 2017-6. Available
# at SSRN. pg. 28.
# Excluded: YouTube, Wikipedia, Facebook, donaldjtrump.com, hilaryclinton.com,
# berniesanders.com, BBC

# Collected manually between 19:49 and 20:03 on 09/11/2017.
n_followers={
    'washingtonpost': 5879061,
    'nytimes':14054412,
    'cnn':28077716,
    'politico':1672053,
    'HuffPost':9055923,
    'TheHill':1249250,
    'realclearpolitics':111872,
    'theguardian':7427957,
    'wsj':5726592,
    'bloombergbusiness':2555756,
    'Breitbart':3363706,
    'ABCNews':11676960,
    'FoxNews':15361413,
    'msnbc':1954638,
    'usatoday':8463043,
    'NBCNews':9034259 ,
    'CBSNews':4315090,
    'Vox':1738051,
    'TheAtlantic':2157917,
    'thedailybeast':2085691,
    'Reuters':3816448,
    'NPR':5788880,
    'latimes':2583657,
    'politifact':775496,
    'BuzzFeed':10094661,
    'yahoonews':7093334,
    'nationalreview':1016928,
    'Slate':1391496,
    'NYPost':3996508,
    'TheWashingtonTimes':586661,
    'DailyCaller':4734098,
    'NYDailyNews':2689562,
    'DailyMail':12300707,
    'businessinsider':7544056,
    'salon':871668,
    'WashingtonExaminer':664562,
    'motherjones':1468370,
    'newyorker':3898225,
    'fivethirtyeight':366507,
    'NewYorkMag':3408054,
    'talkingpointsmemo':391149,
    'forbes':5018417
    }

# Sort based on n followers
rev_dict=dict([ (v,k) for k,v in n_followers.iteritems() ])
for k_ in reversed(sorted(rev_dict.keys())):
    print '{}: {}'.format(rev_dict[k_],k_)
    
