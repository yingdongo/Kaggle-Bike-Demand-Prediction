# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:52:29 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
import numpy as np
from tools import cv_score
from tools import cross_val
from tools import cross_log

def create_xrf():
    return ensemble.ExtraTreesRegressor(n_estimators=500,min_samples_split=4,max_features=1.0)
def create_gbr():
    return ensemble.GradientBoostingRegressor(n_estimators=500,learning_rate=0.1,loss='huber',max_features=0.3)
def create_bagging():
    return ensemble.BaggingRegressor(n_estimators=200)
def create_rf():
    return ensemble.RandomForestRegressor(n_estimators=500,max_features=1.0,min_samples_split=2)

#train=load_data('train.csv')
#data_preprocess(train)
#feature_engineering(train)
#test=load_data('test.csv')
#test_ids=test['datetime']
#data_preprocess(test)
#feature_engineering(test)
#feature_cols=['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'year', 'weekday', 'month', 'hour', 'temp_diff']
#feature_cols=['hour','year','temp','workingday','month','weekday','humidity','atemp','weather','windspeed','temp_diff']#best
#X_train=train[feature_cols]
#y=train['count']
#gbr=create_xrf()#best
#gbr.fit(X_train[feature_cols],y)
#y_count=list(gbr.predict(test[feature_cols]))



def write_submission(filename,test_ids,preds):
    with open(filename, "wb") as outfile:
         outfile.write("datetime,count\n")
         for e, val in enumerate(preds):
             outfile.write("%s,%s\n"%(test_ids[e],np.around(abs(val))))

def get_start():
    train=load_data('train.csv')
    test=load_data('test.csv')
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    score=cv_score(rf,X_train.values,y.values)
    print score#1.24783014078
    #rf.fit(X_train,y)
    #preds=list(rf.predict(test[feature_cols]))
    #write_submission('submissions/count_rf100.csv',test_ids,preds)
    #1.24783014078

def predct_c_r():
    train=load_data('train.csv')
    test=load_data('test.csv')
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train=train[feature_cols]
    casual=train['casual']
    registered=train['registered']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    rf.fit(X_train,casual)
    casual_pred=rf.predict(test[feature_cols])
    rf.fit(X_train,registered)
    registered_pred=rf.predict(test[feature_cols])
    preds=list(abs(casual_pred)+abs(registered_pred))
    write_submission('submissions/casual_registered_rf100.csv',test_ids,preds)
#1.37128
def data_p_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    cv_score(rf,X_train.values,y.values)#0.325312229694
    rf.fit(X_train,y)
    preds=list(rf.predict(test[feature_cols]))
    write_submission('submissions/count_preprogress_rf100.csv',test_ids,preds)
    # 0.47144
    
def feature_eng_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    score=cross_val(rf,X_train,y,day)#cross_val cv_score 0.336275575003
    print score.mean()#0.336297801716
    #rf.fit(X_train,y)
    #preds=list(rf.predict(test[feature_cols]))
    #write_submission('submissions/count_engineering_rf100.csv',test_ids,preds)
    # 0.47709
def log_count_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    score=cross_log(rf,X_train,y,day)
    print score.mean()#0.32295204621
   # rf.fit(X_train,np.log(y+1))
   # preds=list(np.exp(rf.predict(test[feature_cols]))-1)
   # write_submission('submissions/log_count_solution_rf100.csv',test_ids,preds)
    #0.43880
def log_count_selection_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['month','caompare_temp','day','datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    rf=ensemble.RandomForestRegressor(n_estimators=100)
    score=cross_log(rf,X_train,y,day)
    print score.mean()#0.3212452496 without month 0.332202233379
    rf.fit(X_train,np.log(y+1))
    preds=list(np.exp(rf.predict(test[feature_cols]))-1)
    write_submission('to_submit/log_count_selection_solution_rf100_month.csv',test_ids,preds)
    #0.43645
def log_count_param_solution():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=data_preprocess(train)
    test=data_preprocess(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    day=train['day']
    test_ids=test['datetime']
    feature_cols= [col for col in train.columns if col  not in ['month','caompare_temp','day','datetime','count','casual','registered']]
    X_train=train[feature_cols]
    y=train['count']
    #bagging=ensemble.BaggingRegressor(n_estimators=200,max_features=1.0,max_samples=0.8)#0.43511
    ##[0.31665970269152988, 'Bagging', {'max_features': 1.0, 'max_samples': 0.8, 'n_estimators': 200}]
#[0.29591614564520363, 'gbr', {'max_features': 0.8, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 500}]#0.41876
    ##[0.29426222139202357, 'gbr', {'max_features': 0.7, 'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 5}]
#[0.31536901686641666, 'random forest', {'max_features': 0.8, 'min_samples_split': 6, 'n_estimators': 600}]
#[0.31441464727759555, 'extra trees', {'max_features': 1.0, 'min_samples_split': 10, 'n_estimators': 1100}]

    #reg=ensemble.GradientBoostingRegressor(n_estimators=800,learning_rate=0.03,max_features=0.7,max_depth=5)#0.300004022202 #0.41503 without month 0.37466
    #reg=ensemble.RandomForestRegressor(n_estimators=600,min_samples_split=6,max_features=0.8)#0.322599590102   #0.42462 0.39193
    reg=ensemble.ExtraTreesRegressor(n_estimators=1100,min_samples_split=10,max_features=1.0)#0.318951726272
   #0.40683 without month 0.38601
    #ensemble 0.37175
    score=cross_log(reg,X_train,y,day)
    print score.mean()
    #reg.fit(X_train,np.log(y+1))
    #preds=list(np.exp(reg.predict(test[feature_cols]))-1)
    #write_submission('to_submit/log_count_param_solution_gbr800.3.75_month.csv',test_ids,preds)
    
def main():
#    log_count_solution()
    log_count_selection_solution()
if __name__ == '__main__':
    main()