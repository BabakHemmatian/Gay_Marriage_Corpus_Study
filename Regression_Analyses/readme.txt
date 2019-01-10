This folder includes, R code, data files and results associated with regression analyses reported in the manuscript.

Ratings includes the human ratings of impactful comments used to develop the regression models.

words-cons is a list with the best word-based predictors of consequentialist discourse (see the manuscript or null_model.py and associated notes for how this list was derived). words-vb is a similar file for protected-values-based discourse.

Expert-SameSex-no4-02-ratings contains an R data structure with human ratings of impactful comments and how they relate to LDA's topics

Expert-SameSex-no4-02-objs contains an R data structure with the results of all one hundred regression analyses performed on the data mentioned above. It includes correlations with ratings, adjusted R-squared values, coefficients and classification accuracy for each analysis.

LDA is an R-markdown file containing the code used to produce the two data structures mentioned above.

Expert-SameSex-no4-words-01-ratings contains an R data structure with human ratings of impactful comments and how they relate to the word-frequency-based model described in the manuscript.

Expert-SameSex-no4-words-01-objs contains an R data structure with the results of all one hundred regression analyses performed on the data mentioned for the word-based model. It includes correlations with ratings, adjusted R-squared values, coefficients and classification accuracy for each analysis.

Word-based is an R-markdown file containing the code used to produce the two data structures mentioned above.

Figures_and_poly_trends is an R markdown file with the code used to produce the figures provided in the manuscript and to run the polynomial regression analysis for significance of top topic trends that are reported in the paper. It requires Data_for_R to work properly.

The following files are stored in the following directory instead due to large size: 

data_for_R: Includes a summary of topic assignments for every word in every document formatted for use with R code.

The following files are included because they are needed for certain obsolete parts of the code to run properly, but should be otherwise ignored:
* popular_comments_50
* sample_keys
* sample_ratings