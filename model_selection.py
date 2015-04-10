# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:57:02 2015

@author: Ying
"""
from sklearn import ensemble
from tools import load_data
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess 
from feature_selection import cv_score
from feature_selection import split_data1
from feature_selection import get_features
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression



def create_rg():
    models=[]
    models.append
    models.append(('linearRg',LinearRegression()))
    models.append(('ElasticNet',ElasticNet(),))
    models.append(('Lasso',Lasso()))
    models.append(('linearSVR',SVR(kernel='linear')))
    models.append(('rbfSVR',SVR(kernel='rbf')))
    models.append(('AdaBooost',ensemble.AdaBoostRegressor()))
    models.append(('AdaBooost',ensemble.AdaBoostRegressor()))
    models.append(('Bagging',ensemble.BaggingRegressor()))
    models.append(('ExtraTrees',ensemble.ExtraTreesRegressor()))
    models.append(('GB',ensemble.GradientBoostingRegressor()))
    models.append(('RandomForest',ensemble.RandomForestRegressor()))
    return models

def clf_score(models,X_train,y_train):
    index=[]
    score=[]
    for clf in models:
        index.append(clf[0])
        score.append(cv_score(clf[1],X_train,y_train))
    return pd.DataFrame(score,index=index)

def main():
    train=load_data('train.csv')
    train=data_preprocess(train)
    train=feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['datetime','count','casual','registered']]
    X_train,y=split_data1(train,feature_cols)
    cols=get_features(X_train,y,10)
    rg_scores=clf_score(create_rg(),X_train[cols],y)
    print rg_scores
#==============================================================================
#     X_train,y_train1,y_train2=split_data(train,feature_cols)
#     cols1=get_features(X_train,y_train1,11)
#     clf_scores1=clf_score(create_rg(),X_train[cols1],y_train1)
#     print clf_scores1
#     plt.plot(clf_scores1)
#     plt.title('casual')
#     plt.xticks(range(len(clf_scores1)), clf_scores1.index, fontsize=14, rotation=90)
#     plt.show()
# 
#     cols2=get_features(X_train,y_train2,9)
#     clf_scores2=clf_score(create_rg(),X_train[cols2],y_train2)
#     print clf_scores2
#     plt.plot(clf_scores2)
#     plt.title('registered')
#     plt.xticks(range(len(clf_scores2)), clf_scores2.index, fontsize=14, rotation=90)
#==============================================================================
    #plt.show()

if __name__ == '__main__':
    main()