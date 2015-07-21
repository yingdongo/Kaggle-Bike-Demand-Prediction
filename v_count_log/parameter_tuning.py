# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:56:19 2015

@author: Ying
"""
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from explore_data import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
from feature_selection import split_data
from sklearn.grid_search import ParameterGrid
from tools import cross_val
import pandas as pd
def get_rgs():
    return {
            'gbr' : { 
                'est' :ensemble.GradientBoostingRegressor(),
                'grid' : {
                'loss' :['ls', 'huber','lad','quantile'],
                'n_estimators' : [50,200,500],
                'learning_rate': [.1,.03,.01],
                'max_features': [1.0, .3, .1],
                }
            },
            'Bagging' : { 
                'est' :ensemble.BaggingRegressor(),
                'grid' : {
                'n_estimators' : [10,50,200],
                'max_features': [1.0, .3, .1],
                'max_samples': [1.0, .3, .1],
                }
            },
            'extra trees' : { 
                'est' :ensemble.ExtraTreesRegressor(),
                'grid' : {
                'n_estimators' : [50,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,4,6],
                }
            },
            'random forest' : { 
                'est' :ensemble.RandomForestRegressor(),
                'grid' : {
                'n_estimators' : [50,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,4,6],
                }
            },
        }
def get_rgs1():
    return {
            'gbr' : { 
                'est' :ensemble.GradientBoostingRegressor(),
                'grid' : {
                'loss' :['ls', 'huber','lad','quantile'],
                'n_estimators' : [400,500,600],
                'learning_rate': [.1,.03],
                'max_features': [1.0, .9,.8]
                }
            },
            'Bagging' : { 
                'est' :ensemble.BaggingRegressor(),
                'grid' : {
                'n_estimators' : [100,200,300],
                'max_features': [1.0, .9, .8],
                'max_samples': [1.0, .9, .8]
                }
            },
            'extra trees' : { 
                'est' :ensemble.ExtraTreesRegressor(),
                'grid' : {
                'n_estimators' : [800,1000,1100],
                'max_features': [1.0, .9, .8],
                'min_samples_split':[6,7,8]
                }
            },
            'random forest' : { 
                'est' :ensemble.RandomForestRegressor(),
                'grid' : {
                'n_estimators' : [400,500,600],
                'max_features': [1.0, .9, .8],
                'min_samples_split':[6,7,8]
                }
            },
        }
def get_rgs2():
    return {
#==============================================================================
#             'gbr' : { 
#                 'est' :ensemble.GradientBoostingRegressor(),
#                 'grid' : {
#                 'n_estimators' : [800,1000],
#                 'learning_rate': [.03,0.01],
#                 'max_features': [1.,.8,.7],
#                 'max_depth':[5,6,7]
#                 }
    #         },
#==============================================================================
#==============================================================================
#                 'extra trees' : { 
#                 'est' :ensemble.ExtraTreesRegressor(),
#                 'grid' : {
#                 'n_estimators' : [1000,1100],
#                 'max_features': [1.0, .7],
#                 'min_samples_split':[8,9,10]
#                 },
    #       }
#==============================================================================
                'random forest' : { 
                'est' :ensemble.RandomForestRegressor(),
                'grid' : {
                'n_estimators' : [1000,1100],
                'max_features': [1.0],
                'min_samples_split':[7,8,9]
                }
            }
         }
def grid_search(X_train,y,clfs,day):
    scores=pd.DataFrame(columns=['score', 'name','params'])
    for name,clf in clfs.iteritems(): 
        print name 
        param_grid=clfs[name]['grid']
        param_list = list(ParameterGrid(param_grid))
        for i in range(0,len(param_list)):
           reg=clfs[name]['est'].set_params(**param_list[i])
           cv=cross_val(reg,X_train,y,day)
           scores.append([cv.mean(),name,param_list[i]])
           print [cv.mean(),name,param_list[i]]
    return scores
    
def main():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    day=train['day']
    feature_cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    grid_search(X_train[feature_cols],y,get_rgs2(),day)

    
if __name__ == '__main__':
    main()

#[0.31967633399752915, 'Bagging', {'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 200}]
#[0.29850899476671544, 'gbr', {'max_features': 1.0, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 500}]
#[0.31806956786956608, 'random forest', {'max_features': 1.0, 'min_samples_split': 6, 'n_estimators': 500}]
#[0.31588264112447206, 'extra trees', {'max_features': 1.0, 'min_samples_split': 6, 'n_estimators': 1000}]

#[0.31665970269152988, 'Bagging', {'max_features': 1.0, 'max_samples': 0.8, 'n_estimators': 200}]
#[0.29591614564520363, 'gbr', {'max_features': 0.8, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 500}]

#[0.29426222139202357, 'gbr', {'max_features': 0.7, 'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 5}]

#[0.31458698036345151, 'extra trees', {'max_features': 1.0, 'min_samples_split': 8, 'n_estimators': 1000}]
#[0.31536901686641666, 'random forest', {'max_features': 0.8, 'min_samples_split': 6, 'n_estimators': 600}]

#[0.31441464727759555, 'extra trees', {'max_features': 1.0, 'min_samples_split': 10, 'n_estimators': 1100}]
