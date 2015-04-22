# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:57:02 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import cv_score
from feature_selection import split_data1
from feature_selection import get_features
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

class Reg:
    def __init__(self, r0, r1):
        self.r0 = r0
        self.r1 = r1
    def fit(self, xs, ys):
        self.r0.fit(xs, np.log(ys['registered'] + 1))
        self.r1.fit(xs, np.log(ys['casual'] + 1))
    def predict(self, xs):
        ys0 = np.exp(self.r0.predict(xs)) - 1
        ys1 = np.exp(self.r1.predict(xs)) - 1
        ys = np.around(ys0 + ys1)
        ys[ys < 0] = 0
        return ys

def cross_val(reg, X,Y,day):
    print 'cross validation...'
    scores = []
    # chose the continuous date as test set as (10,11), (11,12), ... (18, 19)
    # close to the real station
    for d in range(10, 19):
        test = np.logical_or(day == d, day== (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, tr_y)
        y = reg.predict(tt_x)       
        score = mean_squared_error(np.log(y + 1), np.log(np.around(tt_y['registered'] + tt_y['casual'] + 1))) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)
    
def cross_val1(reg, X,Y,day):
    #print 'cross validation...'
    scores = []
    # chose the continuous date as test set as (10,11), (11,12), ... (18, 19)
    # close to the real station
    for d in range(10, 19):
        test = np.logical_or(day == d, day== (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, tr_y)
        y = reg.predict(tt_x)       
        score = mean_squared_error(y, tt_y) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)


def create_rg():
    models=[]
    models.append
    models.append(('linearRg',LinearRegression()))
    models.append(('ElasticNet',ElasticNet(),))
    models.append(('Lasso',Lasso()))
    models.append(('linearSVR',SVR(kernel='linear')))
    models.append(('rbfSVR',SVR(kernel='rbf')))
    models.append(('AdaBooost',ensemble.AdaBoostRegressor()))
    models.append(('AdaBooost',ensemble.AdaBoostRegressor()))
    models.append(('Bagging',ensemble.BaggingRegressor()))
    models.append(('ExtraTrees',ensemble.ExtraTreesRegressor()))
    models.append(('GB',ensemble.GradientBoostingRegressor()))
    models.append(('RandomForest',ensemble.RandomForestRegressor()))
    return models

def clf_score(models,X_train,y_train,day):
    index=[]
    score=[]
    for clf in models:
        index.append(clf[0])
        score.append(cross_val1(clf[1],X_train,y_train,day).mean())
    return pd.DataFrame(score,index=index)

def main():
    train=load_data('train.csv')
    train=data_preprocess(train)
    train=feature_engineering(train)
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data1(train,feature_cols)
    #cols=get_features(X_train,y,10)
    rg_scores=clf_score(create_rg(),X_train[feature_cols],np.log(y + 1),day)
    print rg_scores

train=load_data('train.csv')
train=data_preprocess(train)
train=feature_engineering(train)
day=train['day']
feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
X_train,y=split_data1(train,feature_cols)
#cols=get_features(X_train,y,10)
reg0= ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)
reg=ensemble.BaggingRegressor(base_estimator=reg0,n_estimators=10)
rg_scores=cross_val1(reg,X_train[feature_cols],np.log(y + 1),day)
print rg_scores


#if __name__ == '__main__':
#    main()