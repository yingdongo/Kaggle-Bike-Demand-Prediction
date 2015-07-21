# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:57:02 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from tools import cross_val
from feature_selection import split_data
import pandas as pd

import numpy as np

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




def create_rg():
    models=[]
    models.append(('Bagging',ensemble.BaggingRegressor()))
    models.append(('ExtraTrees',ensemble.ExtraTreesRegressor()))
    models.append(('GB',ensemble.GradientBoostingRegressor()))
    models.append(('RandomForest',ensemble.RandomForestRegressor()))
    return models
    
def create_rg1():
    models=[]
    models.append(('Bagging',ensemble.BaggingRegressor()))
    models.append(('ExtraTrees',ensemble.ExtraTreesRegressor()))
    models.append(('GB',ensemble.GradientBoostingRegressor()))
    models.append(('RandomForest',ensemble.RandomForestRegressor()))
    return models
def clf_score(models,models1,X_train,y_train,day):
    index=[]
    score=[]
    for clf in models:
        for clf1 in models1:
            index.append(clf[0]+clf1[0])
            cv=cross_val(Reg(clf[1],clf1[1]),X_train,y_train,day).mean()
            print clf[0]+" "+clf1[0]
            print cv
            score.append(cv)
    return pd.DataFrame(score,index=index)

def main():
    train=load_data('train.csv')
    train=data_preprocess(train)
    train=feature_engineering(train)
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    #cols=get_features(X_train,y,10)
    rg_scores=clf_score(create_rg(),create_rg1(),X_train[feature_cols],y,day)
    print rg_scores



if __name__ == '__main__':
    main()

#BaggingBagging            0.328482
#BaggingExtraTrees         0.331503
#BaggingGB                 0.325462
#BaggingRandomForest       0.328731
#ExtraTreesBagging         0.329073
#ExtraTreesExtraTrees      0.335750
#ExtraTreesGB              0.329123
#ExtraTreesRandomForest    0.336549
#GBBagging                 0.352721
#GBExtraTrees              0.355041
#GBGB                      0.361221
#GBRandomForest            0.353482
#RandomForestBagging       0.327145
#RandomForestExtraTrees    0.330594
#RandomForestGB            0.325422
#RandomForestRandomForest  0.324407