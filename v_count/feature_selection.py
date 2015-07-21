# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:18:28 2015

@author: Ying
"""
from sklearn import ensemble
from sklearn import cross_validation
import numpy as np
from matplotlib import pyplot as plt
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from tools import cv_score
import pandas as pd
def split_data(data,cols):
    X=data[cols]
    y=data['count']
    return X,y

def create_rf():
    forest=ensemble.RandomForestRegressor()
    return forest

def feature_importances(rg,X_train,y_train):
    rg.fit(X_train,y_train)
    return rg.feature_importances_
  


def plot_importances(importances, col_array):
# Calculate the feature ranking
    indices = np.argsort(importances)[::-1]    
#Mean Feature Importance
    print "\nMean Feature Importance %.6f" %np.mean(importances)    
#Plot the feature importances of the forest
    plt.figure(figsize=(20,8))
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices],
            color="gr", align="center")
    plt.xticks(range(len(importances)), col_array[indices], fontsize=14, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    print importances[indices]
    
def select_feature(X_train,y_train,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=len(importances)
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        score[i-f_start]=cv_score(create_rf(),X_train[cols].values,y_train.values)
    return pd.DataFrame(score,index=f_range)

def select_by_rf():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    importances=feature_importances(create_rf(),X_train,y)
    plot_importances(importances,X_train.columns)
    score=select_feature(X_train,y,X_train.columns,importances)
    print score
    plt.figure(figsize=(20,8))
    plt.plot(score)
    plt.title('features for count')
    plt.xticks(range(len(score)),score.index)
    plt.show()
def select_feature1(X_train,y):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X_train, np.log(np.around(y)+1))
    return selector
def select_rfecv():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    estimator = SVR(kernel="linear")
    selector=RFECV(estimator, step=1, cv=5)
    selector=selector.fit(X_train, y)
    print selector.support_ 
    print selector.ranking_
    print selector.grid_scores_
    print feature_cols
    
def main():
    select_by_rf() 
    #select_rfecv()
#['hour','year','temp','month','workingday','dayofweek','atemp','humidity','weather','temp_diff','windspeed','season','holiday','compare_temp']
#[ 0.61234506  0.08924774  0.08735438  0.04631075  0.04400216  0.03658491
#  0.0286018   0.01861201  0.01453776  0.00811523  0.00761085  0.00495388
#  0.00172348]
#    select_rfecv()
#[1 1 1 1 1 1 1 2 1 1 1 1 1]
#['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'weekday', 'month', 'hour', 'temp_diff']
if __name__ == '__main__':
   main()

#==============================================================================
# 3   0.692324
# 4   0.691552
# 5   0.417093
# 6   0.370442
# 7   0.360027
# 8   0.342719
# 9   0.333825
# 10  0.339209
# 11  0.343587
# 12  0.339477
#==============================================================================
#13  0.341510