# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:57:02 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
from tools import cv_score
from feature_selection import split_data
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def get_features():
    return ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'temp_diff', 'month', 'hour']
    
def create_rg():
    models=[]
    models.append
    models.append(('linearRg',LinearRegression()))
    models.append(('ElasticNet',ElasticNet(),))
    models.append(('Lasso',Lasso()))
    models.append(('linearSVR',SVR(kernel='linear')))
    models.append(('rbfSVR',SVR(kernel='rbf')))
    models.append(('AdaBooost',ensemble.AdaBoostRegressor()))
    models.append(('Bagging',ensemble.BaggingRegressor()))
    models.append(('ExtraTrees',ensemble.ExtraTreesRegressor()))
    models.append(('GB',ensemble.GradientBoostingRegressor()))
    models.append(('RandomForest',ensemble.RandomForestRegressor()))
    return models

def clf_score(models,X_train,y_train):
    index=[]
    score=[]
    for clf in models:
        index.append(clf[0])
        print clf[0]
        score.append(cv_score(clf[1],X_train.values,y_train.values))
    return pd.DataFrame(score,index=index)

def main():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    rg_scores=clf_score(create_rg(),X_train[feature_cols],y)
    plt.plot(rg_scores)
    plt.title('sum')
    plt.xticks(range(len(rg_scores)), rg_scores.index, fontsize=14, rotation=90)
    


if __name__ == '__main__':
    main()