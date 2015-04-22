# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:18:28 2015

@author: Ying
"""
from sklearn import ensemble
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tools import load_data
from tools import Reg
from tools import cross_val
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
def split_data(data,cols):
    X=data[cols]
    y=data[['casual','registered']]
    return X,y

def create_rf():
    forest=ensemble.RandomForestRegressor()  
    forest1=ensemble.RandomForestRegressor()
    return Reg(forest,forest1)

def feature_importances(rg,X_train,y_train):
    rg.fit(X_train,y_train)
    return rg.feature_importances_
  

def select_feature(X_train,y_train,feature_cols,day,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=len(importances)
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        score[i-f_start]=cross_val(create_rf(),X_train[cols],y_train,day)
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
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    importances=feature_importances(create_rf(),X_train,y)
    score=select_feature(X_train,y,X_train.columns,day,importances)
    print score

    plt.show()

if __name__ == '__main__':
    main()
