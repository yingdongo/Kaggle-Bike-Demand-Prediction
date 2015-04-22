# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:56:19 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import split_data1
from feature_selection import get_features
import numpy as np
from sklearn.grid_search import ParameterGrid
import pandas as pd
from tools import cross_val
from tools import Reg
np.random.seed(0)
def get_rgs():
    return {
            'gbr' : { 
                'est' :[ensemble.GradientBoostingRegressor(),ensemble.GradientBoostingRegressor()],
                'grid' : {
                'n_estimators' : [100,200],
                'max_depth': [5,6,7]
                }
            },
            'random forest' : { 
                'est' :[ensemble.RandomForestRegressor(),ensemble.RandomForestRegressor()],
                'grid' : {
                'n_estimators' : [1100,1500],
                'min_samples_split':[10,11,12]
                }
            },
        }

def get_rgs1():
    return {
            
            'random forest' : { 
                'est' :[ensemble.RandomForestRegressor(),ensemble.RandomForestRegressor()],
                'grid' : {
                'n_estimators' : [1000,1100],
                'min_samples_split':[11,12]
                }
            },
        }




def grid_search(X_train,y,day,clfs):
    scores=pd.DataFrame(columns=['score', 'name','name1','params', 'params1'])
    for name,clf in clfs.iteritems(): 
        for name1,clf1 in clfs.iteritems():
            print name 
            print name1
            param_grid=clfs[name]['grid']
            param_grid1=clfs[name1]['grid']
            param_list = list(ParameterGrid(param_grid))
            param_list1 = list(ParameterGrid(param_grid1))
            for i in range(0,len(param_list)):
               for j in range(0,len(param_list1)):
                   reg0=clfs[name]['est'][0].set_params(**param_list[i])
                   reg1=clfs[name1]['est'][1].set_params(**param_list1[j])
                   reg=Reg(reg0,reg1)
                   cv=cross_val(reg,X_train,y,day)
                   scores.append([cv.mean(),name,name1,param_list[i],param_list1[j]])
                   print [cv.mean(),name,name1,param_list[i],param_list1[j]]

    return scores
    
def main():
    train=load_data('train.csv')
    train=data_preprocess(train)
    train=feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data1(train,feature_cols)
    cols=get_features(X_train,y,12)
    print cols
    grid_search(X_train[cols],y,get_rgs())
#GradientBoostingRegressor
#0.823246929921
#{'max_features': 0.3, 'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 1000}
#==============================================================================
#     X_train,y_train1,y_train2=split_data(train,feature_cols)
#     print 'tuning parameters for casual...'
#     feature_cols=get_features(X_train,y_train1,4)
#     grid_search(X_train[feature_cols],y_train1,get_gbr())
#     #0.546094586614
#     #{'max_features': 0.1, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 200}
# 
#     print 'tuning parameters for registered...'
#     feature_cols=get_features(X_train,y_train2,9)
#     grid_search(X_train[feature_cols],y_train2,get_gbr())
#==============================================================================
#==============================================================================
#0.374689375146
#{'max_features': 1.0, 'loss': 'huber', 'learning_rate': 0.01, 'n_estimators': 1000}
#==============================================================================
train=load_data('train.csv')
train=data_preprocess(train)
train=feature_engineering(train)
day=train['day']
feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
X_train=train[feature_cols]
y=train[['registered','casual']]

scores=grid_search(X_train,y,day,get_rgs())

#if __name__ == '__main__':
#    main()




