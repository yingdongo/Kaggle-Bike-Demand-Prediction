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
def create_bb():
    reg0 = ensemble.BaggingRegressor(n_estimators = 1100, random_state = 0, min_samples_split = 12, oob_score = False, n_jobs = -1)
    reg1 = ensemble.RandomForestRegressor(n_estimators = 1100, random_state = 0, min_samples_split = 12, oob_score = False, n_jobs = -1)
    return Reg(reg0,reg1)
def create_gg():
     reg1 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)
     reg0 = ensemble.GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)
     return Reg(reg0,reg1)   
def create_gg1():
     reg1 = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 6)
     reg0 = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 3,learning_rate=0.1)
     return Reg(reg0,reg1)  
def create_gr():
     reg1=ensemble.RandomForestRegressor(n_estimators=1000, min_samples_split=10)
     reg0=ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=5)
     return Reg(reg0,reg1)
def create_rg():
     reg0=ensemble.RandomForestRegressor(n_estimators=900, min_samples_split=11)
     reg1=ensemble.GradientBoostingRegressor(n_estimators=200,max_depth=5)
     return Reg(reg0,reg1)
def create_bg():
    reg0=ensemble.BaggingRegressor(n_estimators=50)
    reg1=ensemble.GradientBoostingRegressor(n_estimators=500,learning_rate=0.1,max_depth=3)
    return Reg(reg0,reg1)
def get_basicrg():
    reg0=ensemble.RandomForestRegressor(n_estimators= 400)
    reg1=ensemble.RandomForestRegressor(n_estimators= 400)    
    return Reg(reg0,reg1)  

def basicrg_predict(X_train,y,X_test):
    rg=get_basicrg()
    rg.fit(X_train,y)
    y_pre=rg.predict(X_test)
    return y_pre

def rg_predict(rg,X_train,y,X_test):
    rg.fit(X_train,y)
    y_pre=rg.predict(X_test)
    return y_pre

def simple_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    reg0=ensemble.RandomForestRegressor(n_estimators= 100)
    reg1=ensemble.RandomForestRegressor(n_estimators= 100)    
    reg=Reg(reg0,reg1)
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

def engineering_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    reg0=ensemble.RandomForestRegressor(n_estimators= 100)
    reg1=ensemble.RandomForestRegressor(n_estimators= 100)    
    reg=Reg(reg0,reg1)
    
    score=cross_val(reg,X_train,y,day)
    print 'engineering_solution:'
    print score.mean()
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'submissions/rf100_rf100.csv') #0.43499
def selection_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    cols= [col for col in train.columns if col  not in ['month','day','datetime','count','casual','registered']]
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    reg0=ensemble.RandomForestRegressor(n_estimators= 100)
    reg1=ensemble.RandomForestRegressor(n_estimators= 100)    
    reg=Reg(reg0,reg1)
    score=cross_val(reg,X_train,y,day)
    print 'selection_solution:'
    print score.mean()#0.326101952211
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'to_submit/rf100_rf100_month.csv')
    
def paramtuning_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    cols= [col for col in train.columns if col  not in ['month','day','datetime','count','casual','registered']]
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    X_train=train[cols]
    y=train[['registered','casual']]
    X_test=test[cols]
    test_ids=test['datetime']
    #[0.29657889049550584, 'gbr1', 'gbr', {'n_estimators': 500, 'max_depth': 3}, {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}]
    #reg0=ensemble.GradientBoostingRegressor(n_estimators= 500,max_depth=3)
    #reg1=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)    
    #0.37399
    
    #[0.30412708909046249, 'gbr1', 'extra trees', {'n_estimators': 500, 'max_depth': 3}, {'min_samples_split': 10, 'n_estimators': 1000}]
    #reg0=ensemble.GradientBoostingRegressor(n_estimators= 500,max_depth=3)
    #reg1=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=10)    
    
    #[0.30175291564177587, 'gbr', 'gbr', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}]
    #reg0=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)#0.302298687188
    #reg1=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5) #0.36899
    
    #[0.31188731027854039, 'extra trees', 'gbr1', {'min_samples_split': 9, 'n_estimators': 1000}, {'n_estimators': 500, 'max_depth': 3}]
    #reg0=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=9)        
    #reg1=ensemble.GradientBoostingRegressor(n_estimators= 500,max_depth=3)
    
    #[0.29793958119927855, 'gbr', 'random forest', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'min_samples_split': 8, 'n_estimators': 500}]
    #reg0=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)
    #reg1=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=8)   #0.302564934886 

    #[0.31052237683458117, 'extra trees', 'random forest', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 6, 'n_estimators': 500}]
    #reg0=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)#0.316701989476    
    #reg1=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=6)    
   
    #[0.31698795457988432, 'extra trees', 'extra trees', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 10, 'n_estimators': 1000}]
    #reg0=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)    
    #reg1=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=10)    #0.317153927279
    
    #[0.32037254768863044, 'random forest', 'random forest', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 8, 'n_estimators': 500}]
    #reg0=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
    #reg1=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=8) #0.321115940898
    
    #[0.3066113932919306, 'extra trees', 'gbr', {'min_samples_split': 10, 'n_estimators': 900}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
    #reg0=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)    #0.313161123986
    #reg1=ensemble.GradientBoostingRegressor(n_estimators= 800,learning_rate=0.03,max_depth=3)
    #[0.29783062205390975, 'gbr', 'extra trees', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'min_samples_split': 9, 'n_estimators': 1000}]
    #reg0=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)
    #reg1=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=10) #0.302394242383   

    #[0.30727972714289437, 'random forest', 'gbr', {'min_samples_split': 8, 'n_estimators': 600}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
    #reg0=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
    #reg1=ensemble.GradientBoostingRegressor(n_estimators= 800,learning_rate=0.03,max_depth=8) #
    #[0.31090319098589592, 'random forest', 'extra trees', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 10, 'n_estimators': 900}]
    reg0=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
    reg1=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)#0.321018320502
    
    reg=Reg(reg0,reg1)

    score=cross_val(reg,X_train,y,day)
    print 'paramtuning_solution:'
    print score.mean()
    y_pred=rg_predict(reg,X_train,y,X_test)
    write_result(test_ids,y_pred,'to_submit/paramtuning_solution_rf6008_ex90010_month.csv')


def final_solution(reg,train,test):
    test_ids=[]
    y_pred=[]
    write_result(test_ids,y_pred,'final_solution.csv')
    
def write_result(test_ids,y,filename):
    with open(filename, "wb") as outfile:
         outfile.write("datetime,count\n")
         for e, val in enumerate(y):
             outfile.write("%s,%s\n"%(test_ids[e],abs(val)))

#['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'weekday', 'hour', 'temp_diff']
#==============================================================================
# 
# train=load_data('train.csv')
# test=load_data('test.csv')
# train=data_preprocess(train)
# test=data_preprocess(test)
# train=feature_engineering(train)
# test=feature_engineering(test)
# cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
# cols=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'year', 'weekday', 'hour', 'temp_diff']
# X_train=train[cols]
# y=train[['casual','registered']]
# X_test=test[cols]
# test_ids=test['datetime']
# reg=create_gg1()
# y_pred=rg_predict(reg,X_train,y,X_test)
# write_result(test_ids,y_pred,'bgrt1006_gbrt5003_without_monthandwind.csv')
#==============================================================================

def main():
    #selection_solution()
    paramtuning_solution()
if __name__ == '__main__':
    main()