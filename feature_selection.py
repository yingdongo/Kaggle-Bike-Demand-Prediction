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
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
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
    rf=ensemble.RandomForestRegressor(n_estimators=1000, min_samples_split=6, oob_score=True)
    return rf

def feature_importances(rg,X_train,y_train):
    rg.fit(X_train,y_train)
    return rg.feature_importances_
  
def cv_score(rg,X_train,y_train):
    score=cross_validation.cross_val_score(rg, X_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score.mean()

def select_feature(X_train,y_train,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=len(importances)
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        score[i-f_start]=cv_score(create_rf(),X_train[cols],y_train)
    return pd.DataFrame(score,index=f_range)


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
    train=data_preprocess(train)
    train=feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]

    X_train,y=split_data1(train,feature_cols)
    importances=feature_importances(create_rf(),X_train,y)
    score=select_feature(X_train,y,X_train.columns,importances)
    print score
#==============================================================================
#     X_train,y_train1,y_train2=split_data(train,feature_cols)
#     importances1=feature_importances(create_rf(),X_train,y_train1)
#     plot_importances(importances1,X_train.columns)
#     score1=select_feature(X_train,y_train1,X_train.columns,importances1)
#     print score1
#     plt.figure(figsize=(20,8))
#     plt.plot(score1)
#     plt.title('features for casual')
#     plt.xticks(range(len(score1)),score1.index)
#     plt.show()
# 
#     importances2=feature_importances(create_rf(),X_train,y_train2)
#     plot_importances(importances1,X_train.columns)
#     score2=select_feature(X_train,y_train2,X_train.columns,importances2)
#     print score2
#     plt.figure(figsize=(20,8))
#     plt.plot(score2)
#     plt.title('features for registered')
#     plt.xticks(range(len(score2)),score2.index)
#==============================================================================
    plt.show()

if __name__ == '__main__':
    main()
