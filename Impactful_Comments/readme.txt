popular_comments_50 shows the information associated with each of the 2000 most impactful posts. Topic contributions are from the LDA with 50 topics (reported in the manuscript). Please ask contributors for files with topic contributions according to different numbers of topics.

files with exp_pop_comm_ in their names each identify a set of comments chosen to be rated by a separate rater. The choice is based on topic contributions as documented in popular_comments_proc. The first column is the index in that document, the second column the text of the comments

Ratings includes human ratings obtained for all 8 sets (demographics come at the end):
* comp_ questions refer to comprehension checks. All raters passed the checks.
* Comment_DIGIT(DIGIT) variables determine whether a certain comment has been seen by the rater associated with each row (1 means seen). The first digit determines the relevant set, while the second digit determines the index of the comment within that set.
* Comment_Pro_DIGIT(DIGIT) variables determine whether the relevant comment was pro or against same-sex marriage (1 is pro, 2 is against, 3 is I can't tell)
* Comment_Src_DIGIT(DIGIT) variables determine the ratings along the dimension of interest (1 is completely protected-values-based, 4 is neither, 7 is completely consequentialist)

mturk_data-expert.pkl is a pickled file containing summary statistics of rated impactful comments

NOTE: These files should be in the main directory for the functions to run properly. They have been moved to a separate directory to increase readability of the repository
