# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:52:29 2015

@author: Ying
"""
from sklearn import ensemble
from explore_data import load_data
from feature_engineering import add_feature
from feature_selection import split_data1
from feature_selection import split_data
from feature_selection import get_features

def create_gbr_c():
    return ensemble.GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,loss='huber',max_features=0.1)
def create_gbr_r():
    return ensemble.GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01,loss='huber',max_features=1.0)
def create_gbr():
    return ensemble.GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,loss='huber',max_features=0.3)

train=load_data('train.csv')
test=load_data('test.csv')
add_feature(train)
add_feature(test)
feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]

X_train,y=split_data1(train,feature_cols)
X_test=test[feature_cols]
test_ids=test['datetime']
cols=get_features(X_train,y,12)
gbr=create_gbr()
gbr.fit(X_train[cols],y)
y_count=list(gbr.predict(X_test[cols]))

#X_train,y1,y2=split_data(train,feature_cols)
#X_test=test[feature_cols]
#test_ids=test['datetime']
#cols1=get_features(X_train,y1,9)
#cols2=get_features(X_train,y2,11)
# 
#gbr1=create_gbr_c()
#gbr1.fit(X_train[cols1],y1)
#y_c=list(gbr1.predict(X_test[cols1]))
#gbr2=create_gbr_r()
#gbr2.fit(X_train[cols2],y2)
#y_r=list(gbr2.predict(X_test[cols2]))

with open('submit1.csv', "wb") as outfile:
     outfile.write("datetime,count\n")
     for e, val in enumerate(y_count):
         outfile.write("%s,%s\n"%(test_ids[e],abs(val)))