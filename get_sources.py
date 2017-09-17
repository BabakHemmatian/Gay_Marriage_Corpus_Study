# Choose popular political platforms that represent a range of political
# viewpoints.
#
# We ranked each source in Faris et. al.'s list of Top 50 media sources* by 
# media inlinks (pg. 28) by number of Facebook followers. From this ranked
# list, we chose the first 5 sources with a liberal bias**, the first 5 
# sources with a conservative bias**, and the first five sources that we
# classify as middle-of-the-road**.
#
# *From this list we excluded YouTube, Wikipedia, Facebook, donaldjtrump.com, 
# hilaryclinton.com, berniesanders.com, and BBC.
# 
# **"Bias" is calculated using Faris et. al.'s Partisanship score (Appendix 
# 3):
# s = <partisanship score>
# -1<=s<-.10 --> Liberal bias
# -.10<=s<=.10 --> Middle-of-the-road
# .10<s<=1 --> Conservative bias
#
# Reference: Faris, Robert and Roberts, Hal and Etling, Bruce and Bourassa, 
# Nikki and Zuckerman, Ethan and Benkler, Yochai, Partisanship, Propaganda, 
# and Disinformation: Online Media and the 2016 U.S. Presidential Election 
# (August 2017). Berkman Klein Center Research Publication 2017-6. Available
# at SSRN.

# n_followers collected manually between 19:49 and 20:03 on 09/11/2017.
top_media={
    'washingtonpost': { 'n_followers': 5879061, 'p_score': -.51 },
    'nytimes':{ 'n_followers': 14054412, 'p_score': -.53 },
    'cnn':{ 'n_followers': 28077716, 'p_score': -.36 },
    'politico':{ 'n_followers': 1672053, 'p_score': -.29 },
    'HuffPost':{ 'n_followers': 9055923, 'p_score': -.78 },
    'TheHill':{ 'n_followers': 1249250, 'p_score': -.05 },
    'realclearpolitics':{ 'n_followers': 111872, 'p_score': .22 },
    'theguardian':{ 'n_followers': 7427957, 'p_score': -.61 },
    'wsj':{ 'n_followers': 5726592, 'p_score': .05 },
    'bloombergbusiness':{ 'n_followers': 2555756, 'p_score': -.24 },
    'Breitbart':{ 'n_followers': 3363706, 'p_score': .95 },
    'ABCNews':{ 'n_followers': 11676960, 'p_score': -.24 }, # ABC News is 
    # listed twice in Appendix 3. Include only the first listing.
    'FoxNews':{ 'n_followers': 15361413, 'p_score': .87 },
    'msnbc':{ 'n_followers': 1954638, 'p_score': -.67 },
    'usatoday':{ 'n_followers': 8463043, 'p_score': -.19 },
    'NBCNews':{ 'n_followers': 9034259, 'p_score': -.50 },
    'CBSNews':{ 'n_followers': 4315090, 'p_score': -.31 },
    'Vox':{ 'n_followers': 1738051, 'p_score': -.84 },
    'TheAtlantic':{ 'n_followers': 2157917, 'p_score': -.62 },
    'thedailybeast':{ 'n_followers': 2085691, 'p_score': -.72 },
    'Reuters':{ 'n_followers': 3816448, 'p_score': -.20 },
    'NPR':{ 'n_followers': 5788880, 'p_score': -.70 },
    'latimes':{ 'n_followers': 2583657, 'p_score': -.39 },
    'politifact':{ 'n_followers': 775496, 'p_score': -.83 },
    'BuzzFeed':{ 'n_followers': 10094661, 'p_score': -.57 },
    'yahoonews':{ 'n_followers': 7093334, 'p_score': -.15 },
    'nationalreview':{ 'n_followers': 1016928, 'p_score': .54 },
    'Slate':{ 'n_followers': 1391496, 'p_score': -.73 },
    'NYPost':{ 'n_followers': 3996508, 'p_score': .78 },
    'TheWashingtonTimes':{ 'n_followers': 586661, 'p_score': .76 },
    'DailyCaller':{ 'n_followers': 4734098, 'p_score': .88 },
    'NYDailyNews':{ 'n_followers': 2689562, 'p_score': -.63 }, # In Appendix 3
    # Daily News is listed. I'm inferring that the authors are referring to
    # the NY Daily News.
    'DailyMail':{ 'n_followers': 12300707, 'p_score': .71 },
    'businessinsider':{ 'n_followers': 7544056, 'p_score': -.06 },
    'salon':{ 'n_followers': 871668, 'p_score': -.82 },
    'WashingtonExaminer':{ 'n_followers': 664562, 'p_score': .82 },
    'motherjones':{ 'n_followers': 1468370, 'p_score': -.86 },
    'newyorker':{ 'n_followers': 3898225, 'p_score': -.81 },
    'fivethirtyeight':{ 'n_followers': 366507, 'p_score': -.74 },
    'NewYorkMag':{ 'n_followers': 3408054, 'p_score': -.71 },
    'talkingpointsmemo':{ 'n_followers': 391149, 'p_score': -.81 },
    'forbes':{ 'n_followers': 5018417, 'p_score': .02 }
    }

left, middle, right = [],[],[]

def categorize(dict_):
    assert 'name' and 'p_score' in dict_
    name,p_score=dict_['name'],dict_['p_score']
    if -1<=p_score<-.10:
        if len(left)<5:
            left.append(name)
    if -.10<=p_score<=.10:
        if len(middle)<5:
            middle.append(name)
    if .10<p_score<=1:
        if len(right)<5:
            right.append(name)

# Sort based on n followers
by_followers=dict([ (top_media[k]['n_followers'],
                     {'name':k,'p_score':top_media[k]['p_score']}
                    ) for k in top_media.keys() 
                  ])
for k_ in reversed(sorted(by_followers.keys())):
    categorize(by_followers[k_])
