# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:21:54 2015

@author: Ying
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from tools import load_data
from data_preprocess import data_preprocess 
from feature_engineering import feature_engineering
from sklearn.ensemble import RandomForestRegressor

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
    def get_params(self, deep=True):
        out0=self.r0.get_params(self.r0)
        out1=self.r1.get_params(self.r1)
        return [out0,out1]
    def set_params(self, params):
        self.r0.set_params(**params[0])
        self.r1.set_params(**params[1])
        return self

def cross_val(reg, X,Y,day):
    print 'cross validation...'
    scores = []
    # kf = KFold(X.shape[0], 10)
    # for train, test in kf:
    # chose the continuous date as test set as (10,11), (11,12), ... (18, 19)
    # close to the real station
    for d in range(10, 19):
        test = np.logical_or(day == d, day== (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, tr_y)
        y = reg.predict(tt_x)       
        score = mean_squared_error(np.log(y + 1), np.log(np.around(tt_y['registered'] + tt_y['casual'] + 1))) ** 0.5
        print 'score = ', score
        scores.append(score)
    return np.array(scores)

train=load_data('train.csv')
train=data_preprocess(train)
train=feature_engineering(train)
day=train['day']
feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
X_train=train[feature_cols]
y=train[['registered','casual']]
r=RandomForestRegressor()

grid={'min_samples_split': 11, 'n_estimators': 1000}

reg0=r.set_params(**grid)
reg1=r.set_params(**grid)
reg=Reg(reg0,reg1)
score=cross_val(reg,X_train,y,day)