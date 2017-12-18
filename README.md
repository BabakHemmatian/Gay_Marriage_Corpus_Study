# Same-Sex Marriage Reddit Study

### Collaborators: 
### Babak Hemmatian, Steven A. Sloman, Uriel Cohen-Priva
#### Brown University, Department of Cognitive, Linguistic and Psychological Sciences
### Sabina J. Sloman
#### Carnegie Mellon University, Department of Social and Decision Sciences

### Outline

This repository holds Python code that can be used to extract information about trends in Reddit comments that relate to same-sex marriage over the years. Trivial changes to [Utils.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Utils.py) can make it possible to use the code for extracting and analyzing comments related to any topic. 

### Dataset

The code is written to work with [a pre-existing corpus composed of all posts on Reddit from 2006 until the present](http://files.pushshift.io/reddit/comments/). This compressed corpus ignores hierarchical associations between comments. Sample files from the dataset have been included in the repository and can be recognized by the format RC_YYYY_MM.bz2. 

### Topic Modeling

You can set hyperparameters and use Latent Dirichlet Allocation (LDA) to create and examine topic models of the corpus using [Reddit_LDA_Analysis.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Reddit_LDA_Analysis.py). Functions for determining top topics, temporal trends and sampling top comments associated with each topic are included. This file should be in the same directory as [Utils.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Utils.py). 

### Recurrent Neural Networks

You can set hyperparameters and call functions to create recurrent Neural Network (NN) models of the corpus using [Combined_NN_Model.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Combined_NN_Model.py). Based on the hyperparameters, this code can be used to create a language model of the corpus (trained on predicting the upcoming word in a given comment), or a comment classifier. The language model can be used as pre-training for the classifier if training data is scarce. The current default version of the code predicts the sign of each comment's upvotes (negative, neutral, positive). This file also needs to be in the same directory as [Utils.py](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/blob/master/Utils.py).

### Facebook Comments

The collaborators plan to extend this project by including comments from major news outlets on Facebook. The [Facebook](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Facebook) folder holds the unfinished code for scraping and analyzing Facebook comments. [Old_Code](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study/tree/master/Old_Code) also includes unfinished code that is archived only for the use of the collaborators.
