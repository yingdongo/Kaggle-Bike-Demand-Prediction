# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:56:19 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import split_data
from feature_selection import get_features
import numpy as np
from sklearn.grid_search import ParameterGrid
import pandas as pd
from tools import Reg
from tools import cross_val
np.random.seed(0)

def get_rgs():
    return {
#==============================================================================
#             'gbr1' : { 
#                 'est' :[ensemble.GradientBoostingRegressor(),ensemble.GradientBoostingRegressor()],
#                 'grid' : {
#                 'n_estimators' : [500],
#                 'max_depth': [3]
#                 }
#             },
#==============================================================================

#==============================================================================
#             'gbr' : { 
#                 'est' :[ensemble.GradientBoostingRegressor(),ensemble.GradientBoostingRegressor()],
#                 'grid' : {
#                 'n_estimators' : [700,800],
#                 'learning_rate':[0.03],
#                 'max_depth': [3,5,7]
#                 }
#             },
#==============================================================================

             'extra trees' : { 
                 'est' :[ensemble.ExtraTreesRegressor(),ensemble.ExtraTreesRegressor()],
                 'grid' : {
                 'n_estimators' : [900,1000],
                 'min_samples_split':[8,9,10]
                 }
             },

#==============================================================================
#             'random forest' : { 
#                 'est' :[ensemble.RandomForestRegressor(),ensemble.RandomForestRegressor()],
#                 'grid' : {
#                 'n_estimators' : [500,600],
#                 'min_samples_split':[5,6,8],
#                 }
#             },
#==============================================================================
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
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['month','day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    grid_search(X_train,y,day,get_rgs())


if __name__ == '__main__':
    main()
#[0.31883427942976933, 'bagging', 'bagging', {'n_estimators': 50}, {'n_estimators': 10}]
#[0.31886600289060169, 'bagging', 'bagging', {'n_estimators': 50}, {'n_estimators': 50}]
#[0.31199420461131877, 'bagging', 'gbr', {'n_estimators': 50}, {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}]
#[0.31259322396292311, 'bagging', 'gbr', {'n_estimators': 50}, {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}]
#[0.31336751477273439, 'bagging', 'gbr', {'n_estimators': 50}, {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 6}]
#[0.31565071240998116, 'bagging', 'random forest', {'n_estimators': 50}, {'min_samples_split': 2, 'n_estimators': 1000}]
#[0.31595388778143252, 'bagging', 'random forest', {'n_estimators': 50}, {'min_samples_split': 6, 'n_estimators': 500}]
#[0.31543899666688313, 'bagging', 'extra trees', {'n_estimators': 50}, {'min_samples_split': 10, 'n_estimators': 500}]
#[0.31527329042253815, 'bagging', 'extra trees', {'n_estimators': 50}, {'min_samples_split': 10, 'n_estimators': 1000}]
#[0.29791799825545029, 'gbr', 'bagging', {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}, {'n_estimators': 50}]
#[0.29931441632752731, 'gbr', 'bagging', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}, {'n_estimators': 50}]
#[0.29754622099094163, 'gbr', 'gbr', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}, {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 6}]
#[0.29664522493790657, 'gbr', 'gbr', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}, {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}]
#[0.29766333469617162, 'gbr', 'random forest', {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}, {'min_samples_split': 6, 'n_estimators': 500}]
#[0.29768113805089069, 'gbr', 'random forest', {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}, {'min_samples_split': 2, 'n_estimators': 500}]
#[0.29744769813231259, 'gbr', 'extra trees', {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}, {'min_samples_split': 6, 'n_estimators': 500}]
#[0.29740144642113425, 'gbr', 'extra trees', {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}, {'min_samples_split': 10, 'n_estimators': 500}]
#[0.31197527315314721, 'random forest', 'bagging', {'min_samples_split': 10, 'n_estimators': 1000}, {'n_estimators': 50}]



#[0.29793958119927855, 'gbr', 'random forest', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'min_samples_split': 8, 'n_estimators': 500}]
#[0.29783062205390975, 'gbr', 'extra trees', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'min_samples_split': 9, 'n_estimators': 1000}]
#[0.30727972714289437, 'random forest', 'gbr', {'min_samples_split': 8, 'n_estimators': 600}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
#[0.31129049891146432, 'random forest', 'random forest', {'min_samples_split': 8, 'n_estimators': 500}, {'min_samples_split': 8, 'n_estimators': 600}]
##[0.31052237683458117, 'extra trees', 'random forest', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 6, 'n_estimators': 500}]
#[0.31144123954350955, 'extra trees', 'extra trees', {'min_samples_split': 9, 'n_estimators': 900}, {'min_samples_split': 9, 'n_estimators': 1000}]
#[0.3066113932919306, 'extra trees', 'gbr', {'min_samples_split': 10, 'n_estimators': 900}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
#[0.31090319098589592, 'random forest', 'extra trees', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 10, 'n_estimators': 900}]



##[0.30175291564177587, 'gbr', 'gbr', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}]
#[0.30208543535960303, 'gbr', 'gbr1', {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 5}, {'n_estimators': 500, 'max_depth': 3}]
#[0.30406016879708908, 'gbr1', 'gbr', {'n_estimators': 500, 'max_depth': 3}, {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}]
#[0.30438130256545393, 'gbr1', 'gbr1', {'n_estimators': 500, 'max_depth': 3}, {'n_estimators': 400, 'max_depth': 5}]
##[0.30412708909046249, 'gbr1', 'extra trees', {'n_estimators': 500, 'max_depth': 3}, {'min_samples_split': 10, 'n_estimators': 1000}]
#[0.31188731027854039, 'extra trees', 'gbr1', {'min_samples_split': 9, 'n_estimators': 1000}, {'n_estimators': 500, 'max_depth': 3}]
#[0.30517439102116994, 'gbr1', 'random forest', {'n_estimators': 500, 'max_depth': 3}, {'min_samples_split': 5, 'n_estimators': 600}]
##[0.31698795457988432, 'extra trees', 'extra trees', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 10, 'n_estimators': 1000}]
##[0.32037254768863044, 'random forest', 'random forest', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 8, 'n_estimators': 500}]
