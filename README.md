# Box-Office-Prediction

Background:
The projection and analysis of box office performance can be extremely significant for the movie industry and movie fans
Box office performance is a predominant index to measure the achievement of a movie.
Predicting financial success of movies is of great importance to movie industry, as well as to attract funding for future works.

Technology:
Web scraping
Data Preprocessing
Machine Learning(Regression)
Linear Regression
Support Vector Machine
Random Forest
Decision Tree 

Data Source
IMDB: www.imdb.com
Time period: 2010-2013
Features: 19
Numerical: 8
Categorical: 6
Text:  5
Raw data: 31,779 observations


Data Preprocessing
Steps:
Deleting rows
Missing value
Categorical data / Text data
Normalization
Size: 1,653 observations
Training:Validation:Test = 6:2:2


Features
Numerical: duration, release date (3), number of countries, number of languages, budget, revenue
Categorical: genre, content rating, is_USA, is_English, is_3D, is_IMAX
Text: name, director, stars (3)


Model Comparison:
Metrics Used for Evaluation: MAE, MSE, RMSE

Metrics	LR	SVR	RF	DT
MAE	0.0406	0.0569	0.0302	0.0316
MSE	0.0047	0.0073	0.0043	0.0055
RMSE	0.0686	0.0742	0.0652	0.0740

From the above table, we can find that Random Forest Regression is evidently superior to the other two models in terms of MAE, MSE and RMSE comparison. Therefore, Random Forest Regression is our champion model to forecast the box office.


Prediction Result:
R-Square of all test movies: 0.753
This model performs better on blockbusters than other movies.


Discussion & Future Work 
Among all features, we find that duration, star score, is_IMAX, is_3D, number of languages, is_USA, number of countries, content rating and genre have stronger effects on box office than other features.
A 3D, IMAX, action movie, which is casted by famous stars and produced by American movie companies is more likely to be a blockbuster.
Storyline also plays an important role on the success of movies. In the future, we’d like to apply text mining to storyline and include this feature in our prediction model.                                                                            




