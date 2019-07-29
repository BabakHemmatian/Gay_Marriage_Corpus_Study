# Think of the Consequences: A Decade of Discourse About Same-sex Marriage

### Collaborators: 
### Babak Hemmatian
#### Brown University, Department of Cognitive, Linguistic and Psychological Sciences
### Sabina J. Sloman
#### Carnegie Mellon University, Department of Social and Decision Sciences
### Steven A. Sloman, Uriel Cohen Priva
#### Brown University, Department of Cognitive, Linguistic and Psychological Sciences

### Citation
Hemmatian, B., Sloman, S.J., Cohen Priva, U., & Sloman, S.A. (2019). Think of the consequences: A decade of discourse about same-sex marriage. *Behavior Research Methods*. https://doi.org/10.3758/s13428-019-01215-3. 

### Outline

This repository holds Python and R code that can be used to extract information about trends in Reddit comments that relate to same-sex marriage since 2006. Trivial changes to the filter at the end of [lda_defaults.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/lda_defaults.py) can make it possible to use the code for extracting and analyzing comments related to any topic. 

This repository was developed in the process of preparing an academic manuscript published in the peer-reviewed journal *Behavior Research Methods*, accessible [here](https://link.springer.com/article/10.3758/s13428-019-01215-3). Results reported in the manuscript and included in this repository are based on Reddit data from January 2006 to September 2017. 

More details about the contents of each directory in this repository can be found in readme.txt files included in the relevant folder.

### Dataset

The code is written to work with [a pre-existing corpus composed of all posts on Reddit from 2006 until the present](http://files.pushshift.io/reddit/comments/). This compressed corpus ignores hierarchical associations between comments. Sample files from the dataset have been included in the repository and can be in the [Sample_Original_Reddit_Data](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Sample_Original_Reddit_Data) folder. The repository includes some functions for downloading and managing the corpus data. It can be used to retrieve certain high-level statistics, pre-process the data, or draw random equally-large subsamples of comments from different years for further analysis. The outputs of functions are written to file and can be readily imported for future iterations.

Original text of posts in our reported corpus, as well as pre-processed versions of them can be found in the [Corpus](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Corpus) folder.

### Analysis Tools and Results

#### Topic Modeling

You can use Latent Dirichlet Allocation (LDA) to create and examine topic models of the corpus via [Reddit_LDA_Analysis.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Reddit_LDA_Analysis.py). Default hyperparameters for the model to be trained can be set using [lda_defaults.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/lda_defaults.py), and can be overridden by assigning the desired value in [lda_config.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/lda_config.py). Functions for determining top topics via contribution to contextual word-topic assignments, sampling top comments associated with each topic that pass certain criteria, and extracting temporal trends are included. Unless the default path variable is changed, this file should be in the same directory as [Utils.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Utils.py). 

Models developed for the manuscript using this module can be found in the [Learned_Models](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Learned_Models) folder. 

A sample of comments most representative of each topic in the LDA model reported in the manuscript can be found in the [Most_Repr_Comments](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Most_Repr_Comments) folder. The following IDs can be used to identify the reported topics:
* 4 employer attitude and regulations
* 12 religious arguments
* 14 cultural and historical status
* 16 forcing vs. allowing behaviors
* 22 politicians' stance
* 27 children of same-sex parents
* 28 same-sex marriage as a policy issue
* 33 personal anecdotes
* 48 freedom of belief
* 49 LGBT rights

The set of top words associated with models with various numbers of topics (discussed in the manuscript) can be found in [Top_Words](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Top_Words).

Python files with "impactful" in the filename were used to find the most popular and unpopular posts in the corpus (based on upvotes) and gather ratings for their association with two specific classes of arguments (see the manuscript for more details). The set of sampled comments that were used in our study and their associated ratings can be found in the [Impactful_Comments](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Impactful_Comments) folder.

#### Word-based model

You can use nullmodel.py to train a keyword-based binary classifier on a set of rated comments (in the case of our manuscript, ratings of most impactful posts in the corpus). The model uses KL-divergence between word-document co-occurence distributions to find the keywords that best represent one of the two classes.

#### Regression based on topics and keywords

R code was developed to examine the predictive capacity and classification accuracy of LDA and the word-based model for predicting ratings of impactful comments using linear and mixed-effects regressions. The code, as well as associated data and results reported in the manuscript, can be found in [Regression_Analyses](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Regression_Analyses). This folder also includes code for potting the contribution of top topics to the LDA model.

#### Recurrent Neural Networks (in progress)

You can set hyperparameters and call functions to create recurrent Neural Network (NN) models of the corpus using [Combined_NN_Model.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Combined_NN_Model.py). Based on the hyperparameters, this code can be used to create a language model of the corpus (trained on predicting the upcoming word in a given comment), or a comment classifier. The language model can be used as pre-training for the classifier if training data is scarce. The current default version of the code predicts the sign of each comment's upvotes (negative, neutral, positive). This file also needs to be in the same directory as [Utils.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Utils.py). A separate neural network for regression over human interval ratings of comments will be added to the repository.

#### Facebook Comments

The collaborators plan to extend this project by including comments from major news outlets on Facebook. The [Facebook](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Facebook) folder holds the unfinished code for scraping and analyzing Facebook comments. [Old_Code](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Old_Code) also includes unfinished code that is archived for the internal use of the collaborators.
