# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:18:28 2015

@author: Ying
"""
from sklearn import ensemble
import numpy as np
from matplotlib import pyplot as plt
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from tools import cross_val
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
    
def select_feature(X_train,y_train,feature_cols,day,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=len(importances)
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        score[i-f_start]=cross_val(create_rf(),X_train[cols],y_train,day).mean()
    return pd.DataFrame(score,index=f_range)
    
def select_feature1(X_train,y):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X_train, np.log(np.around(y)+1))
    return selector

def select_by_rf():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    importances=feature_importances(create_rf(),X_train,y)
    plot_importances(importances,X_train.columns)
    score=select_feature(X_train,y,X_train.columns,day,importances)
    print score
    plt.figure(figsize=(20,8))
    plt.plot(score)
    plt.title('features for count')
    plt.xticks(range(len(score)),score.index)
    plt.show()
    
def select_rfecv():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    selector=select_feature1(X_train,y)
    print selector.support_ 
    print selector.ranking_
    print selector.grid_scores_
    print feature_cols
def main():
#    select_by_rf()    
#['hour','year','temp','month','atemp','weekday','workingday','humidity','weather','temp_diff','windspeed','season','holiday']
#[ 0.60389575  0.08927242  0.05562972  0.05154929  0.04569275  0.04549952
#  0.04356758  0.02655373  0.01409159  0.00796151  0.00764408  0.00635438
#  0.00228768]
    select_rfecv()
#[ True  True  True  True False  True  True False  True  True  True  True
#  True]
#[1 1 1 1 3 1 1 2 1 1 1 1 1]
#[-0.21041232 -0.18947061  0.09888364  0.21499619  0.23261714  0.24069502
#  0.3186224   0.37348924  0.37418203  0.37386578  0.40171352  0.40043108
#  0.3613429 ]
#['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'weekday', 'month', 'hour', 'temp_diff']
if __name__ == '__main__':
    main()

#3   0.662042
#4   0.706530
#5   0.695409
#6   0.401144
#7   0.385807
#8   0.354975
#9   0.338548
#10  0.340047
#11  0.337670
#12  0.338446