# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:56:19 2015

@author: Ying
"""
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from explore_data import load_data
from feature_engineering import add_feature
from feature_selection import split_data1
from feature_selection import get_features

def get_gbr():
    return {
            'gbr' : { 
                'est' :ensemble.GradientBoostingRegressor(),
                'grid' : {
                'loss' :['ls', 'huber','lad','quantile'],
                'n_estimators' : [50,200,500,1000],
                'learning_rate': [.1,.03,.01],
                'max_features': [1.0, .3, .1],
                }
            },
            'Bagging' : { 
                'est' :ensemble.BaggingRegressor(),
                'grid' : {
                'n_estimators' : [50,200,500,1000],
                'max_features': [1.0, .3, .1],
                'max_samples': [1.0, .3, .1],
                'oob_score':[True,False]
                }
            },
            'extra trees' : { 
                'est' :ensemble.ExtraTreesRegressor(),
                'grid' : {
                'n_estimators' : [50,200,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,4,6],
                'oob_score':[True,False]
                }
            },
            'random forest' : { 
                'est' :ensemble.RandomForestRegressor(),
                'grid' : {
                'n_estimators' : [50,200,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,4,6],
                'oob_score':[True,False]
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
    add_feature(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data1(train,feature_cols)
    cols=get_features(X_train,y,12)
    grid_search(X_train[cols],y,get_gbr())
    #0.366639290392
#{'max_features': 0.3, 'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 200}
#    0.365348643345 for 8 features
#{'max_features': 1.0, 'loss': 'huber', 'learning_rate': 0.03, 'n_estimators': 500}
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

    
if __name__ == '__main__':
    main()




