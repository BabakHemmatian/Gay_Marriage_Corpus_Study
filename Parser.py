from __future__ import print_function
import bz2
from collections import defaultdict, OrderedDict
import datetime
import hashlib
import HTMLParser
import json
import multiprocessing
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import os
from pathlib2 import Path
import pickle
import re
import time
import subprocess
import sys
from config import *
from Utils import *

# This needs to be importable from the main module for multiprocessing
# https://stackoverflow.com/questions/24728084/why-does-this-implementation-of-multiprocessing-pool-not-work
def parse_one_month_wrapper(args):
    year, month, kwargs=args
    Parser(**kwargs).parse_one_month(year, month)

class Parser(object):
    # Parameters:
    #   dates: a list of (year,month) tuples for which data is to be processed
    #   path: Path for data and output files.
    #   stop: List of stopwords.
    #   vote_counting: Include number of votes per comment in parsed file.
    #   NN: Parse for neural net.
    #   write_original: Write a copy of the raw file.
    #   download_raw: If the raw data doesn't exist in path, download a copy from
    #       https://files.pushshift.io/reddit/comments/.
    #   clean_raw: Delete the raw data file when finished.
    def __init__(self, clean_raw=CLEAN_RAW, dates=dates,
                 download_raw=DOWNLOAD_RAW, hashsums=None, NN=NN, path=path,
                 regex=regex, stop=stop, write_original=WRITE_ORIGINAL,
                 vote_counting=vote_counting):
        # check input arguments for valid type
        assert type(vote_counting) is bool
        assert type(NN) is bool
        assert type(write_original) is bool
        assert type(download_raw) is bool
        assert type(clean_raw) is bool
        assert type(path) is str
        # check the given path
        if not os.path.exists(path):
            raise Exception('Invalid path')
        assert type(stop) is set or type(stop) is list

        self.clean_raw=CLEAN_RAW
        self.dates=dates
        self.download_raw=download_raw
        self.hashsums=hashsums
        self.NN=NN
        self.path=path
        self.regex=regex
        self.stop=stop
        self.write_original=write_original
        self.vote_counting=vote_counting

    ## Raw Reddit data filename format
    def _get_rc_filename(self, yr, mo):
        if len(str(mo))<2:
            mo='0{}'.format(mo)
        assert len(str(yr))==4
        assert len(str(mo))==2
        return 'RC_{}-{}.bz2'.format(yr, mo)

    ## Download Reddit comment data
    def download(self, year=None, month=None, filename=None):
        assert not all([ isinstance(year, type(None)),
                         isinstance(month, type(None)),
                         isinstance(filename, type(None))
                       ])
        assert isinstance(filename, type(None)) or (isinstance(year, type(None))
               and isinstance(month, type(None)))
        BASE_URL='https://files.pushshift.io/reddit/comments/'
        if not isinstance(filename, type(None)):
            url=BASE_URL+filename
        else:
            url=BASE_URL+self._get_rc_filename(year, month)
        print ('Sending request to {}.'.format(url))
        os.system('cd {} && wget {}'.format(self.path, url))

    ## Get Reddit compressed data file hashsums to check downloaded files' integrity
    def Get_Hashsums(self):
        # notify the user
        print ('Retrieving hashsums to check file integrity')
        # set the URL to download hashsums from
        url='https://files.pushshift.io/reddit/comments/sha256sum.txt'
        # remove any old hashsum file
        if Path(self.path+'/sha256sum.txt').is_file():
            os.remove(self.path+'/sha256sum.txt')
        # download hashsums
        os.system('cd {} && wget {}'.format(self.path, url))
        # retrieve the correct hashsums
        hashsums = {}
        with open(self.path+'/sha256sum.txt','rb') as f:
            for line in f:
                if line.strip() != "":
                    (val, key) = str(line).split()
                    hashsums[key] = val
        return hashsums

    ## calculate hashsums for downloaded files in chunks of size 4096B
    def sha256(self, fname):
        hash_sha256= hashlib.sha256()
        with open("{}/{}".format(self.path, fname), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _clean(self, text):
        # check input arguments for valid type
        assert type(text) is unicode or type(text) is str

        # remove stopwords --> check to see if apostrophes are properly encoded
        stop_free = " ".join([i for i in text.lower().split() if i.lower() not
                              in self.stop])
        replace = {"should've":"should", "mustn't":"mustn",
                   "shouldn't":"shouldn", "couldn't":"couldn", "shan't":"shan",
                   "needn't":"needn", "-":""}
        substrs = sorted(replace, key=len, reverse=True)
        regexp = re.compile('|'.join(map(re.escape, substrs)))
        stop_free = regexp.sub(lambda match: replace[match.group(0)], stop_free)

        # remove special characters
        special_free = ""
        for word in stop_free.split():
            if "http" not in word and "www" not in word: # remove links
                word = re.sub('[^A-Za-z0-9]+', ' ', word)
                if word.strip() != "":
                    special_free = special_free+" "+word.strip()

        # check for stopwords again
        special_free = " ".join([i for i in special_free.split() if i not in
                                 self.stop])

        return special_free

    ### define the preprocessing function to add padding and remove punctuation, special characters and stopwords (neural network)
    def NN_clean(self, text):

        # check input arguments for valid type
        assert type(text) is list or type(text) is str or type(text) is unicode

        # create a container for preprocessed sentences
        cleaned = []

        for index,sent in enumerate(text): # iterate over the sentences
            special_free = self._clean(sent)

            # add sentence and end of comment padding
            if special_free.strip() != "":
                padded = punc_free+" *STOP*"
                if index+1 == len(text):
                    padded = padded+" *STOP2*"
                cleaned.append(padded)
            elif special_free.strip() == "" and len(text)!=1 and len(cleaned)!=0 and index+1 == len(text):
                cleaned[-1] = cleaned[-1]+" *STOP2*"

        return cleaned

    ### define the preprocessing function to lemmatize, and remove punctuation, special characters and stopwords (LDA)
    # NOTE: Since LDA doesn't care about sentence structure, unlike NN_clean, the entire comment should be fed into this function as a continuous string
    # NOTE: The Reddit dataset seems to encode the quote blocks as just new lines. Therefore, there is no way to get rid of quotes
    def LDA_clean(self, text):
        special_free=self._clean(text)

        # lemmatize
        normalized = " ".join([nltk.stem.WordNetLemmatizer().lemmatize(word) if word != "us" else "us" for word in special_free.split()])

        return normalized

    def get_parser_fns(self, year=None, month=None):
        assert ( ( isinstance(year, type(None)) and isinstance(month, type(None)) ) or
                 ( not isinstance(year, type(None)) and not isinstance(month, type(None)) ) )
        if isinstance(year, type(None)) and isinstance(month, type(None)):
            suffix=""
        else:
            suffix="-{}-{}".format(year, month)
        fns=dict((("lda_prep","{}/lda_prep{}".format(self.path, suffix)),
                  ("original_comm","{}/original_comm{}".format(self.path, suffix)),
                  ("original_indices","{}/original_indices{}".format(self.path, suffix)),
                  ("counts","{}/RC_Count_List{}".format(self.path, suffix)),
                  ("timedict","{}/RC_Count_Dict{}".format(self.path, suffix))
                 ))
        if self.NN:
            fns["nn_prep"]="{}/nn_prep{}".format(self.path, suffix)
        if self.vote_counting:
            fns["votes"]="{}/votes{}".format(self.path, suffix)
        return fns

    # NOTE: Parses for LDA if NN = False
    # NOTE: Saves the text of the non-processed comment to file as well if write_original = True
    def parse_one_month(self, year, month):
        timedict=dict()

        if self.NN: # if parsing for a neural network
            ## import the pre-trained PUNKT tokenizer for determining sentence boundaries
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        decoder = json.JSONDecoder(encoding='utf-8')

        ## prepare files
        filename=self._get_rc_filename(year, month) # get the relevant compressed data file name
        if not filename in os.listdir(self.path) and self.download_raw: # if the file is not available on disk and download is turned on
            self.download(year, month) # download the relevant file
            # check data file integrity and download again if needed
            filesum = self.sha256(filename) # calculate hashsum for the data file on disk
            attempt = 0 # number of hashsum check trials for the current file
            while filesum != self.hashsums[filename]: # if the file hashsum does not match the correct hashsum
                attempt += 1 # update hashsum check counter
                if attempt == 5: # if failed hashsum check three times, ignore the error to prevent an infinite loop
                    print("Failed to pass hashsum check 5 times. Ignoring the error.")
                    break
                # notify the user
                print("Corrupt data file detected")
                print("Expected hashsum value: "+self.hashsums[filename]+"\nBut calculated: "+filesum)
                os.remove(self.path+'/'+filename) # remove the corrupted file
                self.download(year,month) # download it again

        elif not filename in os.listdir(self.path): # if the file is not available, but download is turned off
            print ('Can\'t find data for {}/{}. Skipping.'.format(month, year)) # notify the user
            return

        # open the file as a text file, in utf8 encoding
        fin = bz2.BZ2File(self.path+'/'+filename,'r')

        # Get names of processing files
        fns=self.get_parser_fns(year, month)

        # create a file to write the processed text to
        if self.NN: # if doing NN
            fout = open(fns["nn_prep"],'w')
        else: # if doing LDA
            fout = open(fns["lda_prep"],'w')

        # create a file if we want to write the original comments and their indices to disk
        if self.write_original:
            foriginal = open(fns["original_comm"],'w')
            main_indices = open(fns["original_indices"],'w')

        # if we want to record sign of the votes
        if self.vote_counting:
            # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
            vote = open(fns["votes"], 'w')

        # create a file to store the relevant cummulative indices for each month
        ccount = open(fns["counts"], 'w')

        main_counter=0
        processed_counter=0
        ## read data
        for line in fin: # for each comment
            main_counter += 1 # update the general counter

            # parse the json, and turn it into regular text
            comment = decoder.decode(line)
            original_body = HTMLParser.HTMLParser().unescape(comment["body"])

            # filter comments by relevance to the topic
            if len(self.regex.findall(original_body)):
                ## preprocess the comments
                if self.NN:
                    body = sent_detector.tokenize(original_body) # tokenize the sentences
                    body = self.NN_clean(body) # clean the text for NN
                    if len(body) > 0: # if the comment body is not empty after preprocessing
                        processed_counter += 1 # update the counter
                        # if we want to write the original comment to disk
                        if self.write_original:
                            original_body = original_body.replace("\n","") # remove mid-comment lines
                            original_body = original_body.encode("utf-8") # set encoding
                            print(" ".join(original_body.split()),file=foriginal) # record the original comment
                            print(main_counter,file=main_indices) # record the main index

                        for sen in body: # for each sentence in the comment
                            # remove mid-comment lines and set encoding
                            sen = sen.replace("\n","")
                            sen = sen.encode("utf-8")

                            # print the processed sentence to file
                            print(" ".join(sen.split()), end=" ", file=fout)

                        # ensure that each comment is on a separate line
                        print("\n",end="",file=fout)

                else: # if doing LDA
                    body = self.LDA_clean(original_body) # clean the text for LDA
                    if body.strip() != "": # if the comment is not empty after preprocessing
                        processed_counter += 1 # update the counter

                        # if we want to write the original comment to disk
                        if self.write_original:
                            original_body = original_body.replace("\n","") # remove mid-comment lines
                            original_body = original_body.encode("utf-8") # set encoding
                            print(" ".join(original_body.split()),file=foriginal) # record the original comment
                            print(main_counter,file=main_indices) # record the index in the original files

                        # remove mid-comment lines and set encoding
                        body = body.replace("\n","")
                        body = body.encode("utf-8")

                        # print the comment to file
                        print(" ".join(body.split()), sep=" ",end="\n", file=fout)

                # if we are interested in the sign of the votes
                if self.vote_counting:
                    # write the sign of the vote to file (-1 if negative, 0 if neutral, 1 if positive)
                    print(int(comment["score"]),end="\n",file=vote)

                # record the number of documents by year and month
                created_at = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime('%Y-%m')
                timedict[created_at]=timedict.get(created_at, 0)
                timedict[created_at]+=1

        # write the monthly cummulative number of comments to file
        print(processed_counter,file=ccount)

        # close the files to save the data
        fin.close()
        fout.close()
        if self.vote_counting:
            vote.close()
        if self.write_original:
            foriginal.close()
            main_indices.close()
        ccount.close()
        with open(fns["timedict"], "wb") as wfh:
            pickle.dump(timedict, wfh)

        # timer
        print("Finished parsing "+filename+" at " + time.strftime('%l:%M%p'))

        if self.clean_raw: # if the user wishes compressed data files to be removed after processing
            print ("Cleaning up {}/{}.".format(self.path, filename))
            os.system('cd {} && rm {}'.format(self.path, filename)) # delete the recently processed file

        return

    def pool_parsing_data(self):
        fns=self.get_parser_fns()
        # Initialize an "overall" timedict
        timedict=defaultdict(lambda:0)
        for kind in fns.keys():
            fns_=[ self.get_parser_fns(year, month)[kind] for year, month in
                   self.dates ]
            if kind=="timedict":
                # Update overall timedict with data from each year
                for fn_ in fns_:
                    with open(fn_, "rb") as rfh:
                        minitimedict=pickle.load(rfh)
                        for mo, val in minitimedict.items():
                            timedict[mo]+=val
                with open(fns["timedict"], "w") as wfh:
                    for month,docs in sorted(timedict.iteritems()):
                        print(month+" "+str(docs), end='\n', file=wfh)
                continue
            subprocess.call("cat "+" ".join(fns_)+"> "+fns[kind], shell=True)

    def parse(self):
        # get the correct hashsums to check file integrity
        self.hashsums = self.Get_Hashsums()

        # Parallelize parsing by month
        pool = multiprocessing.Pool(processes=CpuInfo())
        inputs=[ (year, month, self.__dict__) for year, month in self.dates                 ]
        pool.map(parse_one_month_wrapper, inputs)

        # timer
        print("Finished parsing at " + time.strftime('%l:%M%p'))

        # Pool parsing data from all files
        self.pool_parsing_data()

    ### Function to call parser when needed and parse comments
    # TODO: Replace mentions of Vote in this file with mentions of sample_ratings
    def Parse_Rel_RC_Comments(self):
        # if preprocessed comments are available, ask if they should be rewritten
        if (self.NN and Path(self.path+"/nn_prep").is_file()) or (not self.NN and Path(self.path+"/lda_prep").is_file()):
            Q = raw_input("Preprocessed comments are already available. Do you wish to delete them and parse again [Y/N]?")
            if Q == 'Y' or Q == 'y': # if the user wishes to overwrite the comments
                # delete previous preprocessed data
                if self.NN: # for NN
                    os.remove(self.path+"/nn_prep")
                    if Path(self.path+"/votes").is_file():
                        os.remove(self.path+"/votes")

                elif not self.NN: # for LDA
                    os.remove(self.path+"/lda_prep")
                if Path(self.path+"/RC_Count_List").is_file():
                    os.remove(self.path+"/RC_Count_List")
                if Path(self.path+"/RC_Count_Dict").is_file():
                    os.remove(self.path+"/RC_Count_Dict")

                # timer
                print("Started parsing at " + time.strftime('%l:%M%p'))
                self.parse()

            else: # if preprocessed comments are available and the user does not wish to overwrite them
                print("Checking for missing files")

                # check for other required files aside from main data
                missing_files = 0

                if not Path(self.path+"/RC_Count_List").is_file():
                    missing_files += 1
                if self.NN:
                    if not Path(self.path+"/votes").is_file():
                        missing_files += 1

                # if there are missing files, delete any partial record and parse again
                if missing_files != 0:
                    print("Deleting partial record and parsing again")

                    if self.NN: # for NN
                        os.remove(self.path+"/nn_prep")
                        if Path(self.path+"/votes").is_file():
                            os.remove(self.path+"/votes")

                    elif not self.NN: # for LDA
                        os.remove(self.path+"/lda_prep")
                    if Path(self.path+"/RC_Count_List").is_file():
                        os.remove(self.path+"/RC_Count_List")

                    # timer
                    print("Started parsing at " + time.strftime('%l:%M%p'))
                    self.parse()

        else:
            if Path(self.path+"/RC_Count_List").is_file():
                os.remove(self.path+"/RC_Count_List")
            if self.NN:
                if Path(self.path+"/votes").is_file():
                    os.remove(self.path+"/votes")

            # timer
            print("Started parsing at " + time.strftime('%l:%M%p'))
            self.parse()

    ### determine what percentage of the posts in each year was relevant based on content filters
    # NOTE: Requires total comment counts (RC_Count_Total) from http://files.pushshift.io/reddit/comments/
    # NOTE: Requires monthly relevant counts from parser or disk
    def Rel_Counter(self):
        if not Path(self.path+"/RC_Count_List").is_file():
            raise Exception('Cumulative monthly comment counts could not be found')
        if not Path(self.path+"/RC_Count_Total").is_file():
            self.download(filename="monthlyCount.txt")
            os.rename(self.path+"/monthlyCount.txt", self.path+"/RC_Count_Total")

        # load the total monthly counts into a dictionary
        d = {}
        with open(self.path+"/RC_Count_Total",'r') as f:
            for line in f:
                line = line.replace("\n","")
                if line.strip() != "":
                    try:
                        (key, val) = line.split("  ")
                    except ValueError:
                        (key, val) = line.split(" ")
                    d[key] = int(val)

        # calculate the total yearly counts
        total_year = {}
        for keys in d:
            if str(keys[3:7]) in total_year:
                total_year[str(keys[3:7])] += d[keys]
            else:
                total_year[str(keys[3:7])] = d[keys]

        relevant_year, _ = Yearly_Counts(self.path)
        relevant = {}
        for idx,year in enumerate(relevant_year):
            relevant[str(2006+idx)] = year

        # calculate the percentage of comments in each year that was relevant and write it to file
        perc_rel = {}
        rel = open(self.path+"/perc_rel",'a+')
        for key in relevant:
            perc_rel[key] = float(relevant[key]) / float(total_year[key])
        print(sorted(perc_rel.items()),file=rel)
        rel.close

    ### Load, calculate or re-calculate the percentage of relevant comments/year
    def Perc_Rel_RC_Comment(self):
        if Path(self.path+"/perc_rel").is_file(): # look for extant record
            # if it exists, ask if it should be overwritten
            Q = raw_input("Yearly relevant percentages are already available. Do you wish to delete them and count again [Y/N]?")

            if Q == 'Y' or Q == 'y': # if yes
                os.remove(path+"/perc_rel") # delete previous record
                self.Rel_Counter() # calculate again
            else: # if no
                print("Operation aborted") # pass

        else: # if there is not previous record
            self.Rel_Counter() # calculate

    ### Helper functions for select_random_comments
    def get_comment_lengths(self):
        fin=self.path+'/lda_prep'
        with open(fin, 'r') as fh:
            return [ len(line.split()) for line in fh.read().split("\n") ]

    def _select_n(self, n, iterable):
        if len(iterable)<n:
            return iterable
        return np.random.choice(iterable, size=n, replace=False)

    # Parameters:
    #   n: Number of random comments to sample.
    #   years_to_sample: Years to select from.
    #   min_n_comments: Combine all comments from years with less than
    #       min_n_comments comments and select from the combined set. E.g. If
    #       min_n_comments = 5000, since there are less than 5000 (relevant)
    #       comments from 2006, 2007 and 2008, a random sample of n will be drawn
    #       from the pooled set of relevant comments from 2006, 2007 and 2008.
    #       Defaults to 5000.
    #   overwrite: If the sample file for the year exists, skip.
    def select_random_comments(self, n=n_random_comments,
                               years_to_sample=years, min_n_comments=5000,
                               overwrite=OVERWRITE):
        # File to write random comment indices to
        fout='random_indices'
        fcounts='random_indices_count'

        path=self.path+'/'
        fout=path+fout
        fcounts=path+fcounts
        if ( not overwrite and os.path.exists(fout) ):
            print ("{} exists. Skipping. Set overwrite to True to overwrite.".format(fout))
            return

        years_to_sample=sorted(years_to_sample)
        ct_peryear, ct_cumyear=Yearly_Counts(self.path)
        ct_lu=dict((y, i) for i, y in enumerate(years))
        early_years=[ yr for yr in years_to_sample if
                      ct_peryear[ct_lu[yr]]<min_n_comments ]

        # Make sure the early_years actually contains the first years in years, if
        # any. Otherwise the order that indices are written to file won't make any
        # sense.
        assert all([ early_years[i]==early_years[i-1]+1 for i in range(1,
                     len(early_years)) ])
        assert all([ yr==yr_ for yr, yr_ in zip(early_years,
                     years_to_sample[:len(early_years)]) ])

        later_years=[ yr for yr in years_to_sample if yr not in early_years ]

        # Record the number of indices sampled per year
        nixs=defaultdict(int)

        # Get a list of comment lengths, so we can filter by it
        lens=self.get_comment_lengths()

        with open(fout, 'w') as wfh:
            if len(early_years)>0:
                fyear, lyear=early_years[0], early_years[-1]
                start=ct_cumyear[ct_lu[fyear-1]] if fyear-1 in ct_lu else 0
                end=ct_cumyear[ct_lu[lyear]]
                ixs_longenough=[ ix for ix in range(start, end) if lens[ix] >=
                                 min_comm_length ]
                ixs=sorted(self._select_n(n, ixs_longenough))
                for ix in ixs:
                    nixs[years[[ ct>ix for ct in ct_cumyear ].index(True)]]+=1
                assert sum(nixs.values())==len(ixs)
                wfh.write('\n'.join(map(str, ixs)))
                wfh.write('\n')
            for year in later_years:
                start=ct_cumyear[ct_lu[year-1]]
                end=ct_cumyear[ct_lu[year]]
                ixs_longenough=[ ix for ix in range(start, end) if lens[ix] >=
                                 min_comm_length ]
                ixs=sorted(self._select_n(n, ixs_longenough))
                nixs[year]=len(ixs)
                wfh.write('\n'.join(map(str, ixs)))
                wfh.write('\n')

        with open(fcounts, 'w') as wfh:
            wfh.write('\n'.join('{} {}'.format(k, v) for k, v in
                      sorted(nixs.iteritems(), key=lambda kv: kv[0])))
