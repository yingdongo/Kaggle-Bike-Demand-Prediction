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
from sklearn.grid_search import GridSearchCV
np.random.seed(0)
def get_rgs():
    return {
            'gbr' : { 
                'est' :ensemble.GradientBoostingRegressor(),
                'grid' : {
                'n_estimators' : [100,200],
                'max_depth': [5,6,7]
                }
            }
            }

def get_rgs1():
    return {
            
            'random forest' : { 
                'est' :ensemble.RandomForestRegressor(),
                'grid' : {
                'n_estimators' : [1000,1100],
                'min_samples_split':[11,12]
                }
            },
        }



def grid_search(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], n_jobs=16, verbose=0, cv=10)
        clf.fit(X_train,y)
        print clf.score
        print clf.best_score_
        print clf.best_params_
    
def main():
    train=load_data('train.csv')
    train=data_preprocess(train)
    train=feature_engineering(train)
    cols= [col for col in train.columns if col  not in ['day','datetime','count','casual','registered']]
    X_train,y=split_data1(train,cols)
    print cols
    grid_search(X_train[cols],y,get_rgs())

if __name__ == '__main__':
    main()




