# -*- coding: utf-8 -*-
"""
Created on Sat May 02 14:22:43 2015

@author: Jessica
"""

import pandas as pd
import random
import os
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as out_in
import matplotlib.pyplot as pl


#os.chdir('D:\BI\BIA656AStatistical Learning & Analytics\FINAL PROJECT')
#os.getcwd()

df1=pd.read_csv('movie_data_2010.csv')
df2=pd.read_csv('movie_data_2011.csv')
df3=pd.read_csv('movie_data_2012.csv')
df4=pd.read_csv('movie_data_2013_1.csv')
df5=pd.read_csv('movie_data_2013_2.csv')
#df6=pd.read_csv('movie_data_2014.csv')
df=pd.DataFrame()

df=df1.append(df2)
df=df.append(df3)
df=df.append(df4)
df=df.append(df5)
#df=df.append(df6)

#deleting null data
df_new=df[df.REVENUE.notnull()]
df_new=df_new[df_new.DIRECTOR.notnull()]
df_new=df_new[df_new.STAR_1.notnull()]
df_new=df_new.reset_index()

#fill in null "BUDGET" with random numbers
budget_list = list(df_new['BUDGET'])

budget_can = []
for budget in budget_list:
    if budget>0:budget_can.append(budget)

for i in range(len(budget_list)):
    if budget_list[i]>0:continue
    else:
        budget_list[i] = budget_can[random.randint(0,len(budget_can)-1)]

df_new['BUDGET']=budget_list

#use value_counts to find mode
df_new['MONTH'].value_counts()
df_new['DAY'].value_counts()

#fill in null variables with mean and mode
df_new.loc[df_new['DURATION'].isnull(),'DURATION'] =np.mean(df_new['DURATION'])
df_new.loc[df_new['MONTH'].isnull(),'MONTH'] =9
df_new.loc[df_new['DAY'].isnull(),'DAY'] =25
df_new.loc[df_new['GENRE'].isnull(),'GENRE'] ='drama'  

df_new.loc[df_new['IS_ENGLISH'].isnull(),'IS_ENGLISH'] =1
df_new.loc[df_new['LANGUAGE_NUM'].isnull(),'LANGUAGE_NUM'] =1
df_new.loc[df_new['CONTENT_RATING'].isnull(),'CONTENT_RATING'] ='unrated'
df_new.loc[df_new['CONTENT_RATING']=='not rated','CONTENT_RATING'] ='unrated'

genre_maj=['drama','comedy','action']
for i in range(len(df_new['GENRE'])):
    if df_new['GENRE'][i] not in genre_maj:
        df_new['GENRE'][i]='others'

#convert categorical variables into numerical variables

dummies1=pd.get_dummies(df_new.CONTENT_RATING)
df_new=df_new.join(dummies1)

dummies2=pd.get_dummies(df_new.GENRE)
df_new=df_new.join(dummies2)

#read files with names and scores of directors and stars
def loadNames(fname):
    
    names=[]
    scores=[]
    f=open(fname)
    for line in f:
        name,nu,score=line.strip().split('\t')    
        names.append(name)    
        scores.append(int(score))
    f.close()

    return names,scores


DIRECTOR,DIRECTOR_SCORE=loadNames('director.txt')
STAR1,STAR1_SCORE=loadNames('star_1.txt')
STAR2,STAR2_SCORE=loadNames('star_2.txt')
STAR3,STAR3_SCORE=loadNames('star_3.txt')
STAR_SCORE=[]

for i in range(len(STAR1_SCORE)):
    score=STAR1_SCORE[i]+STAR2_SCORE[i]+STAR3_SCORE[i]
    STAR_SCORE.append(score)

#create new variables about scores of directors and stars
df_new['DIRECTOR_SCORE']=DIRECTOR_SCORE
df_new['STAR1_SCORE']=STAR1_SCORE
df_new['STAR2_SCORE']=STAR2_SCORE
df_new['STAR3_SCORE']=STAR3_SCORE
df_new['STAR_SCORE']=STAR_SCORE

df_new['M_D']=((df_new['MONTH']-1)*30+df_new['DAY'])/365

df_new['GENRE'].value_counts()



"""
#outlier detection

IQR=np.percentile(df_new['REVENUE'],75)-np.percentile(df_new['REVENUE'],25)
lower_bound=np.percentile(df_new['REVENUE'],25)-1.5*IQR
upper_bound=np.percentile(df_new['REVENUE'],75)+1.5*IQR

df_new=df_new[df_new['REVENUE']<upper_bound]
"""

#normalize budget and revenue before splitting data
scaler=MinMaxScaler()


df_new['REVENUE2']=scaler.fit_transform(df_new['REVENUE'] )
df_new['BUDGET2']=scaler.fit_transform(df_new['BUDGET'] )
df_new['DURATION2']=scaler.fit_transform(df_new['DURATION'] )



#df_new.to_csv('D:\BIA\BIA656A Statistical Learning & Analytics\FINAL PROJECT\df_new1.csv')
"""
df_new1=df_new[['DURATION','M_D','COUNTRY_NUM','IS_USA','LANGUAGE_NUM','IS_ENGLISH','IS_3D',
'IS_IMAX','BUDGET','g','nc-17','pg','pg-13','r','unrated','DIRECTOR_SCORE','STAR_SCORE','drama','comedy','action','others','REVENUE']]
df_new2=df_new[['DURATION','M_D','COUNTRY_NUM','IS_USA','LANGUAGE_NUM','IS_ENGLISH','IS_3D',
'IS_IMAX','BUDGET','CONTENT_RATING','DIRECTOR_SCORE','STAR_SCORE','GENRE','REVENUE']]
df_new1=scaler.fit_transform(df_new1)
df_new2=scaler.fit_transform(df_new2)
"""
L=len(df_new['REVENUE'])
N=range(0,L,1)
np.random.shuffle(N)

train_set=df_new.ix[N[:int(0.6*L)]]
validate_set=df_new.ix[N[int(0.6*L):int(0.8*L)]]
test_set=df_new.ix[N[int(0.8*L):]]
#select x and y
x_train1=train_set[['DURATION2','M_D','COUNTRY_NUM','IS_USA','LANGUAGE_NUM','IS_ENGLISH','IS_3D',
'IS_IMAX','BUDGET2','g','nc-17','pg','pg-13','r','unrated','DIRECTOR_SCORE','STAR_SCORE','drama','comedy','action','others']]


x_validate1=validate_set[['DURATION2','M_D','COUNTRY_NUM','IS_USA','LANGUAGE_NUM','IS_ENGLISH','IS_3D',
'IS_IMAX','BUDGET2','g','nc-17','pg','pg-13','r','unrated','DIRECTOR_SCORE','STAR_SCORE','drama','comedy','action','others']]


x_test1=test_set[['DURATION2','M_D','COUNTRY_NUM','IS_USA','LANGUAGE_NUM','IS_ENGLISH','IS_3D',
'IS_IMAX','BUDGET2','g','nc-17','pg','pg-13','r','unrated','DIRECTOR_SCORE','STAR_SCORE','drama','comedy','action','others']]


y_train=train_set['REVENUE2']
y_validate=validate_set['REVENUE2']
y_test=test_set['REVENUE2']


#build models using decision tree regression, random forest regression, SVR 
#and stochastic gradient descent regression
model1=DecisionTreeRegressor()
model2=RandomForestRegressor()
model3=SVR()
model4=SGDRegressor(loss="squared_loss")

#fit the models to x_train1
model1.fit(x_train1,y_train)
model2.fit(x_train1,y_train)
model3.fit(x_train1,y_train)
model4.fit(x_train1,y_train)


pred11=model1.predict(x_validate1)
pred12=model2.predict(x_validate1)
pred13=model3.predict(x_validate1)
pred14=model4.predict(x_validate1)




test_pred11=model1.predict(x_test1)
test_pred12=model2.predict(x_test1)
test_pred13=model3.predict(x_test1)
test_pred14=model4.predict(x_test1)

#calculate mean absolute error

MAE11=mean_absolute_error(y_validate,pred11)
MAE12=mean_absolute_error(y_validate,pred12)
MAE13=mean_absolute_error(y_validate,pred13)
MAE14=mean_absolute_error(y_validate,pred14)

print "===========Mean Absolute Error=========="
print 'DecisionTree1: ', format(MAE11, '.4f')
print 'RandomForest1: ', format(MAE12, '.4f')
print 'SVR1: ', format(MAE13, '.4f')
print 'SGDRegressor1: ', format(MAE14, '.4f')
print '\n'



MSE11=mean_squared_error(y_validate,pred11)
MSE12=mean_squared_error(y_validate,pred12)
MSE13=mean_squared_error(y_validate,pred13)
MSE14=mean_squared_error(y_validate,pred14)

print "==========Mean Square of Error=========="
print 'OLS (After backward feature selection): ', format(MSE_OLS, '.4f')
print 'DecisionTree1: ', format(MSE11, '.4f')
print 'RandomForest1: ', format(MSE12, '.4f')
print 'SVR1: ', format(MSE13, '.4f')
print 'SGDRegressor1: ', format(MSE14, '.4f')
print '\n'

#calculate root mean squared error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
    

RMSE11=rmse(pred11,y_validate)
RMSE12=rmse(pred12,y_validate)
RMSE13=rmse(pred13,y_validate)
RMSE14=rmse(pred14,y_validate)

print "==========Root Mean Square of Error=========="
print 'DecisionTree1: ', format(RMSE11, '.4f')
print 'RandomForest1: ', format(RMSE12, '.4f')
print 'SVR1: ', format(RMSE13, '.4f')
print 'SGDRegressor1: ', format(RMSE14, '.4f')
print '\n'
  
#calculate coefficient of determination   
def R_S(predicted_value):
    mean_test=np.mean(y_test)
    SSR=np.sum((predicted_value-mean_test)**2)
    SST=np.sum((y_test-mean_test)**2)
    r_s=SSR/SST
    return r_s
    
   
R_S11=R_S(pred11)
R_S12=R_S(test_pred12)
print R_S12
R_S13=R_S(pred13)
R_S14=R_S(pred14)

print "==========R Square==========\n",
print 'DecisionTree1: ', format(R_S11, '.4f')
print 'RandomForest1: ', format(R_S12, '.4f')
print 'SVR1: ', format(R_S13, '.4f')
print 'SGDRegressor1: ', format(R_S14, '.4f')
print '\n'








print '==========Scatter Plot=========='
predlist=[test_pred_ols,test_pred11,test_pred12,test_pred13,test_pred14]
title_list = ['OLS (After backward feature selection)','DecisionTree','RandomForest','SVR','SGDRegressor']
MAE_list = [MAE_OLS,MAE11,MAE12,MAE13,MAE14]
MSE_list = [MSE_OLS,MSE11,MSE12,MSE13,MSE14]
RMSE_list = [RMSE_OLS,RMSE11,RMSE12,RMSE13,RMSE14]
R_list = [R_S_ols,R_S11,R_S12,R_S13,R_S14]

"""
#scatter plots of predicted value versus true value
for i in range(len(predlist)):
    pl.plot(y_test,predlist[i],'ro')
    pl.plot(y_test,y_test,'g')
#    pl.title(title_list[i] + '\n MAE: ' + str(format(MAE_list[i], '.4f')) + \
#    '\n MSE: ' + str(format(MSE_list[i], '.4f'))+'\n RMSE: '+str(format(RMSE_list[i], '.4f'))+'\n R Square: ' + str(format(R_list[i], '.4f')))
    pl.xlabel('true value') 
    pl.ylabel('predictive value')
    pl.show()
"""

#scatter plots of predicted value versus true value

y_test_n=np.log(y_test)
test_pred_ols_n=np.log(test_pred_ols)

test_pred11_n=np.log(test_pred11)
test_pred12_n=np.log(test_pred12)
test_pred13_n=np.log(test_pred13)
test_pred14_n=np.log(test_pred14)



#plot for linear regression
fig,ax=pl.subplots()
pl.title("Linear Regression")
ax.scatter(y_test_n, test_pred_ols_n)
ax.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'k', lw=2)
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
pl.show()

#plot for decision tree regression
fig,ax=pl.subplots()
pl.title("Decision Tree Regression")
ax.scatter(y_test_n, test_pred11_n)
ax.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'k', lw=2)
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
pl.show()

#plot for random forest regression
pl.clf()
pl.figure(figsize=(32,18))
pl.title("Random Forest Regression",fontsize=60)
pl.scatter(y_test_n, test_pred12_n,s=80)
pl.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'k', lw=6)
pl.xlabel('True Value',fontsize=45)
pl.ylabel('Predicted Value',fontsize=45)

pl.show()



#plot for SVR
fig,ax=pl.subplots()
pl.title("Support Vector Machine Regression")
ax.scatter(y_test_n, test_pred13_n)
ax.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'k', lw=2)
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
pl.show()


