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
        score = mean_squared_error(y, np.log(np.around(tt_y)+1)) ** 0.5
        #print 'score = ', score
        scores.append(score)
    return np.array(scores)
