# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:52:29 2015

@author: Ying
"""
from sklearn import ensemble
from feature_selection import split_data
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
import numpy as np
def create_gbr_c():
    return ensemble.GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,loss='huber',max_features=0.1)
def create_xrf():
    return ensemble.ExtraTreesRegressor(n_estimators=500,min_samples_split=4,max_features=1.0)
def create_gbr():
    return ensemble.GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,loss='huber',max_features=0.3)
def create_bagging():
    return ensemble.BaggingRegressor(n_estimators=200)

train=load_data('train.csv')
data_preprocess(train)
feature_engineering(train)
test=load_data('test.csv')
test_ids=test['datetime']
data_preprocess(test)
feature_engineering(test)
#feature_cols=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'year', 'weekday', 'month', 'hour', 'temp_diff']
feature_cols=['hour','year','temp','workingday','month','weekday','humidity','atemp','weather','windspeed','temp_diff']
#feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
X_train,y=split_data(train,feature_cols)
gbr=create_xrf()
gbr.fit(X_train[feature_cols],np.log(y+1))
y_count=list(np.exp(gbr.predict(test[feature_cols]))-1)


with open('submit_extratrees_rf.csv', "wb") as outfile:
     outfile.write("datetime,count\n")
     for e, val in enumerate(y_count):
         outfile.write("%s,%s\n"%(test_ids[e],np.around(abs(val))))