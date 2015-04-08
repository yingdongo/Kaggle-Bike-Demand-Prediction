# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:18:28 2015

@author: Ying
"""
from sklearn import ensemble
from sklearn import cross_validation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from explore_data import load_data
from feature_engineering import add_feature
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
def split_data(data,cols):
    X=data[cols]
    y1=data['casual']
    y2=data['registered']
    return X,y1,y2

def split_data1(data,cols):
    X=data[cols]
    y=data['count']
    return X,y

def create_rf():
    forest=ensemble.GradientBoostingRegressor()
    return forest

def feature_importances(rg,X_train,y_train):
    rg.fit(X_train,y_train)
    return rg.feature_importances_
  
def cv_score(rg,X_train,y_train):
    score=cross_validation.cross_val_score(rg, X_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score.mean()

def select_feature(X,y):
    estimator = SVR(kernel='linear')
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    return selector




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
    
def get_features(X_train,y_train,n):
    importances=feature_importances(create_rf(),X_train,y_train)
    indices = np.argsort(importances)[::-1]   
    cols=X_train.columns[indices]
    return cols[:n]

def main():
    train=load_data('train.csv')
    add_feature(train)
    print train.describe()
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]

    X_train,y=split_data1(train,feature_cols)
    selector=select_feature(X_train,y)
    print selector.support_ 
    print selector.ranking_
    print selector.grid_scores_
    print feature_cols
   #['season', 'workingday', 'weather', 'temp', 'atemp', 'humidity','weekday', 'month', 'hour']

if __name__ == '__main__':
    main()
