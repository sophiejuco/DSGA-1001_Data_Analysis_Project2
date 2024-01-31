# DSGA-1001_Data_Analysis_Project2
DSGA-1001 Data Analysis Project 2: Analysis on movie ratings data set
<br>
This repository contains the code notebook and data used for this project as well as the results of our analysis.
* <em>IDS_Project2.ipynb</em> is the Python notebook.
* <em>movieReplicationSet.csv</em> is the movie ratings data used.
<br>

There are also a folder:<br>
* <em>Analysis plots</em> contains the plots and tables used to show the results of the analysis.
<br>

# Analysis Report
**Question 1:**
The first step is to clean the data. I removed all the rows with missing values and replaced those with the mean of the user’s average rating and the movie’s average rating. Then for each of the 400 movies, I performed 399 simple linear regressions. The movie that produces the highest coefficient of determination is recorded as the best predictor. 

The average COD for all movies is 0.4238. Below is a histogram of the COD for each movie:

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ1_1.png)

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ1_2.png)

**Question 2:**
We added in three additional predictors in Question 2: gender identity, sibship status and social viewing preferences. 

To clean up the data, we removed rows where there are NaNs, and gender identity = 3 (self-described), and those that didn't respond to sibship status and social viewing preferences.
For the 10 best predicted movies, we did a multiple regression using the top predictor’s movie rating, gender identity, sibship status and social viewing preferences. Then we compared the COD from simple linear regression and the COD from multiple linear regression. 

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ2_1.png)

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ2_2.png)

As we can see from the table above, the COD doesn’t change much, so we can conclude that using all 4 predictors doesn’t provide a better estimate than using only the top predictor’s movie rating. 

**Question 3:**
We first take the 30 movies in the middle of the COD range excluding the movies used in question 2, then randomly select 10 movies from all other movies excluding the 30 movies selected or movies used in question 2. The ten movies picked are: Gladiator (2000), Tropic of Cancer (1970), Hollow Man (2000), Speed 2: Cruise Control (1997), Father's Day (1997), Ocean's Eleven (2001), American History X (1998), Midnight Cowboy (1969), Elf (2003), Mulholland Dr. (2001). We then use the ridgeCV function to find the optimal alpha provided a list of possible alphas by cross-validation. The Alpha, RMSE, and betas are given in the table below. 

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ3_1.png)

From this table, we can see most movies have a high alpha value, a higher alpha indicates a lower magnitude of the beta coefficients. 

A higher beta means this movie has a greater impact on predicting the target movie. In ridge regression, the betas can approach 0 but can’t be 0. 

From the RMSE, we see overall the ridge regression predicts the movie well. 

**Question 4:**
For this question, we are using the same movies from question 3, but instead of using ridge regression, we choose to use LASSO regression. Similarly, we pick the optimal Alpha using the LassoCV function. The Alpha, RMSE, and betas are given in the table below. 

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ4_1.png)

With lasso regression, we can reduce the number of predictors. In this table, we can see for 27 out of 30 movies, we see at least one beta turning to 0. 

Compared to ridge regression, the difference in RMSE is very small. 7 movies have lower RMSE in lasso regression, so we can conclude although there is not a significant difference in performace, ridge regression does slightly better here. 

**Question 5:**
The four target movies were found by averaging the ratings of each movie over all viewers, ranking them in ascending order of average ratings, and then pulling the middle four movies from that ranking (indexes 198-202 out of 0-399). The four target movies were found to be Fahrenheit 9/11 (2004), Happy Gilmore (1996), Diamonds are Forever (1971), and Scream (1996). The X data are the average movie ratings per viewer from the non-imputed data with the remaining NaN values being filled with 0 since those viewers did not rate any movies. The y data were found using a median split of the imputed data for each of the four target movies to determine who did not enjoy (score=0 for ratings less than the median) and who did enjoy (score=1 for ratings equal or greater than the median) the movies. 

For each movie, the X data and the generated y data of that movie were split with a third of the data used as testing and the remaining amount used as training data. The random_state was set to 0 for the data split and logistic regression so the resulting data split and beta and AUC scores will remain the same every time the code is run. 
For each movie, sklearn’s LogisticRegressionCV function was used to initialize a logistic regression model with cross validation to prevent overfitting. For each movie, the logistic regression was fitted to the X_train and y_train data and predictions were generated using the predict() operation and the X_test data. Beta values were found using the coef_ operation on the model. The AUC score was calculated using sklearn’s roc_auc_score function with the y_test data and the prediction probabilities of the model (found using the models predict_proba operation using the X_test data). 

For Fahrenheit 9/11 (2004), the beta value is 9.148 and the AUC score is 0.957 which means approximately 96% of the predictions were correct. For Happy Gilmore (1996), the beta value is 0.522 and the AUC score is 0.9196 which means approximately 92% of the predictions were correct. For Diamonds are Forever (1971), the beta value is 8.685 and the AUC score is 0.963 which means approximately 96% of the predictions were correct. For Scream (1996), the beta value is 0.491 and the AUC score is 0.899 which means approximately 90% of predictions were correct.

![alt_tect](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotQ5_1.png)

For all four of these movies, despite varying beta values, using the averaged movie ratings for each user as X produced a logistic regression model that had roughly 90% or higher AUC scores. Most prediction errors occurred around the inflection point which is around the median rating value. This means that viewers who tend to rate movies lower altogether, rated these four movies lower (i.e. enjoyment=0), and viewers who tend to rate movies higher altogether rated these four movies higher (i.e. enjoyment=1).

**Extra Credit:**
 Can viewers’ ratings of “Gets nervous easily” be used as a good parameter/predictor of their enjoyment of a scary movie - The Exorcist (1973)? 

The ratings data (y) for The Exorcist was pulled from the imputed data and was used to generate enjoyment of the movie using a median split (0 for ratings lower than the median, 1 for ratings higher than the median). The nervousness data (X) was pulled from the non-imputed data and any NaN values were filled in with the mean rating of “Gets nervous easily”. The X and y data were split using sklearn’s train_test_split function with a third of the data used as testing and the rest as training. The random_state was set to 0 for the data split and logistic regression so the resulting split and beta and AUC scores will remain the same every time the code is run.

Sklearn’s LogisticRegressionCV function was used to create a logistic regression with cross validation to avoid overfitting. The model was fit to the X_train and y_train data and predictions were generated with the X_test data. The beta value and AUC score were found using the same methods as in Question 5. 

The beta value is -0.00077 and the AUC score is 0.483 which means that the model predicted less than 50% of the values correctly. With such a low beta value and a low AUC score, it is clear that “Gets nervous easily” is not a good predictor of enjoyment of the scary movie The Exorcist.

![alt_text](https://github.com/sophiejuco/DSGA-1001_Data_Analysis_Project2/blob/main/Analysis%20plots/plotEC.png)



