# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:43:25 2015

@author: Ying
"""
from tools import load_data
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from data_preprocess import data_preprocess
from feature_engineering import feature_engineering
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import ensemble
from tools import cv_score
import pandas as pd
train=load_data('train.csv')
train=data_preprocess(train)
train=feature_engineering(train)
day=train['day']
count=train['count']
casual=train['casual']
registered=train['registered']
train=train.drop(['datetime','count','casual','registered','day'],axis=1)

enc = OneHotEncoder(categorical_features=np.array([0,3,9,11,11]))
train_onehot=enc.fit_transform(train).toarray()

train_onehot=pd.DataFrame(train_onehot)

#l_svr=SVR(kernel='linear')#0.95631725022
#score1=cv_score(l_svr,train_onehot.values,count.values)

#r_svr=SVR(kernel='rbf')#1.36135578192
#score2=cv_score(r_svr,train_onehot.values,count.values)

#lr=LinearRegression()#1.03053462633
#score3=cv_score(lr,train_onehot.values,count.values)