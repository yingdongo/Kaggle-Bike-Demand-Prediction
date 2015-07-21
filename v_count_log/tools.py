# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:40:28 2015

@author: Ying
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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
        reg.fit(tr_x, np.log(np.around(tr_y)+1))
        y = reg.predict(tt_x)       
        score = mean_squared_error(y, np.log(tt_y+1)) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)
