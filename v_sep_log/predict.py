# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:52:29 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import get_features
from tools import cross_val
from tools import Reg
import numpy as np

def create_rr():
    reg0 = ensemble.RandomForestRegressor(n_estimators = 1100, random_state = 0, min_samples_split = 12, oob_score = False, n_jobs = -1)
    reg1 = ensemble.RandomForestRegressor(n_estimators = 1100, random_state = 0, min_samples_split = 12, oob_score = False, n_jobs = -1)
    return Reg(reg0,reg1)
def create_gg():
     reg1 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)
     reg0 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)
     return Reg(reg0,reg1)   
def create_gr():
     reg1=ensemble.RandomForestRegressor(n_estimators=1000, min_samples_split=10)
     reg0=ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=5)
     return Reg(reg0,reg1)
def create_rg():
     reg0=ensemble.RandomForestRegressor(n_estimators=900, min_samples_split=11)
     reg1=ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=5)
     return Reg(reg0,reg1)
def get_basicrg():
    reg=ensemble.RandomForestRegressor(n_estimators= 400)
    return reg  
def get_rg():
    reg=ensemble.RandomForestRegressor(n_estimators= 400)
    return reg  
def get_gbtr():
    reg = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 7, random_state = 0)
    return reg
def basicrg_predict(X_train,y,X_test):
    y_pre=get_basicrg().fit(X_train,y).predict(X_test)
    return y_pre
def rg_predict(rg,X_train,y,X_test):
    y_pre=rg.fit(X_train,y).predict(X_test)
    return y_pre

def simple_solution(reg,train,test):
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    score=cross_val(reg,X_train,y)
    print 'preprocess_solution:'
    print score
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'preprocess_solution.csv')

 
def preprocess_solution(reg,train,test):
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    train=data_preprocess(train)
    test=data_preprocess(test)
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    score=cross_val(reg,X_train,y)
    print 'preprocess_solution:'
    print score
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'preprocess_solution.csv')

def engineering_solution(reg,train,test):
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    
    score=cross_val(reg,X_train,y)
    print 'preprocess_solution:'
    print score
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'engineering_solution.csv')
    
def selection_solution(reg,train,test):
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    score=cross_val(reg,X_train,y)
    print 'preprocess_solution:'
    print score
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'selection_solution.csv') 
    
def paramtuning_solution(reg,train,test):
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    score=cross_val(reg,X_train,y)
    print 'preprocess_solution:'
    print score
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'paramtuning_solution.csv')
    

def final_solution(reg,train,test):
    test_ids=[]
    y_pred=[]
    write_result(test_ids,y_pred,'final_solution.csv')
    
def write_result(test_ids,y,filename):
    with open(filename, "wb") as outfile:
         outfile.write("datetime,count\n")
         for e, val in enumerate(y):
             outfile.write("%s,%s\n"%(test_ids[e],abs(val)))


train=load_data('train.csv')
test=load_data('test.csv')
train=data_preprocess(train)
test=data_preprocess(test)
train=feature_engineering(train)
test=feature_engineering(test)
cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
X_train=train[cols]
y=train['count']
X_test=test[cols]
test_ids=test['datetime']
reg=get_gbtr()
day=train['day']
y_pred=rg_predict(reg,X_train,y,X_test)
write_result(test_ids,y_pred,'gbrt_count.csv')