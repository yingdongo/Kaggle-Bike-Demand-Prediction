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

def get_gbr():
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

def grid_search(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], n_jobs=16, verbose=0, cv=10)
        clf.fit(X_train,y)
        print clf.score
        print clf.best_score_
        print clf.best_params_

def main():
    train=load_data('train.csv')
    data_preprocess(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data(train,feature_cols)
    grid_search(X_train[feature_cols],y,get_gbr())

    
if __name__ == '__main__':
    main()

#BaggingRegressor
#0.804544861328
#{'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 200}
#GradientBoostingRegressor
#0.797099618586
#{'max_features': 0.3, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 500}
#RandomForestRegressor
#0.809387050226
#{'max_features': 1.0, 'min_samples_split': 2, 'n_estimators': 500}
#ExtraTreesRegressor
#0.844328270306
#{'max_features': 1.0, 'min_samples_split': 4, 'n_estimators': 500}

