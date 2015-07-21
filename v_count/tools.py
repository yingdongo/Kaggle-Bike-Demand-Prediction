# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:40:28 2015

@author: Ying
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
def load_data(url):
    data = pd.read_csv(url)
    return data

def calculate_rmsle(ypred, yactual):
    N = len(yactual)
    lsum = 0.0
    for i in range(N):
        x, y = ypred[i]+1.0, yactual[i]+1.0
        if x < 1:
            x = 1
        lsum += (np.log(x)-np.log(y))**2
    lsum /= N
    return np.sqrt(lsum)

def cv_score(reg, X,Y):
    results=[]
    skf=cross_validation.StratifiedKFold(Y,n_folds=10, random_state=42,shuffle=True)
    for train_index, test_index in skf:
        X_train, y = X[train_index], Y[train_index]
        fit= reg.fit(X_train, y)
        preds=abs(fit.predict(X[test_index]))
        results.append(calculate_rmsle(Y[test_index], preds) )
    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )
    return np.array(results).mean()

def cross_val(reg, X,Y,day):
    #print 'cross validation...'
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
        y = abs(reg.predict(tt_x))    
        score = mean_squared_error(np.log(np.around(y) + 1), np.log(tt_y + 1)) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)

def cross_log(reg, X,Y,day):
    #print 'cross validation...'
    scores = []
    # kf = KFold(X.shape[0], 10)
    # for train, test in kf:
    # chose the continuous date as test set as (10,11), (11,12), ... (18, 19)
    # close to the real station  
    for d in range(10, 19):
        test = np.logical_or(day == d, day== (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, np.log(tr_y+1))
        y = reg.predict(tt_x)
        score = mean_squared_error(y, np.log(tt_y+1)) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)

