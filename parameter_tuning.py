# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:56:19 2015

@author: Ying
"""
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import split_data1
from feature_selection import get_features
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.grid_search import ParameterGrid
import pandas as pd
np.random.seed(0)
def get_rgs():
    return {
            'gbr' : { 
                'est' :[ensemble.GradientBoostingRegressor(),ensemble.GradientBoostingRegressor()],
                'grid' : {
                'loss' :['ls', 'huber','lad','quantile'],
                'n_estimators' : [100,500,1000],
                'learning_rate': [.1,.03,.01],
                'max_features': [1.0, .3, .1],
                'max_depth': [2,6,10]
                }
            },
            'extra trees' : { 
                'est' :[ensemble.ExtraTreesRegressor(),ensemble.ExtraTreesRegressor],
                'grid' : {
                'n_estimators' : [100,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,6,10]
                }
            },
            'random forest' : { 
                'est' :[ensemble.RandomForestRegressor(),ensemble.RandomForestRegressor()],
                'grid' : {
                'n_estimators' : [100,500,1000],
                'max_features': [1.0, .3, .1],
                'min_samples_split':[2,6,10],
                'oob_score':[True,False]
                }
            },
        }

class Reg:
    def __init__(self, r0, r1):
        self.r0 = r0
        self.r1 = r1
    def fit(self, xs, ys):
        self.r0.fit(xs, np.log(ys['registered'] + 1))
        self.r1.fit(xs, np.log(ys['casual'] + 1))
    def predict(self, xs):
        ys0 = np.exp(self.r0.predict(xs)) - 1
        ys1 = np.exp(self.r1.predict(xs)) - 1
        ys = np.around(ys0 + ys1)
        ys[ys < 0] = 0
        return ys
from sklearn.metrics import mean_squared_error

def cross_val(reg, X,Y,day):
    print 'cross validation...'
    scores = []
    # kf = KFold(X.shape[0], 10)
    # for train, test in kf:
    # chose the continuous date as test set as (10,11), (11,12), ... (18, 19)
    # close to the real station
    for d in range(10, 19):
        test = np.logical_or(day == d, day== (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, tr_y)
        y = reg.predict(tt_x)       
        score = mean_squared_error(np.log(y + 1), np.log(np.around(tt_y['registered'] + tt_y['casual'] + 1))) ** 0.5
        print 'score = ', score
        scores.append(score)
    return np.array(scores)


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
                   reg1=clfs[name1]['est'][1].set_params(**param_list[j])
                   reg=Reg(reg0,reg1)
                   print param_list[i]
                   print param_list[j]
                   print name
                   print name1
                   cv=cross_val(reg,X_train,y,day)
                   scores.append([cv.mean(),name,name1,param_list[i],param_list1[j]])
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




