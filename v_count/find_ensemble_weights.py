# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:52:14 2015

@author: Ying
"""


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from feature_engineering import feature_engineering
from data_preprocess import data_preprocess
train = pd.read_csv("train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
train=data_preprocess(train)
train=feature_engineering(train)
feature_cols= [col for col in train.columns if col  not in ['month','caompare_temp','day','datetime','count','casual','registered']]
X_train=train[feature_cols]

labels = train['count']

print(train.head())

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
### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss=StratifiedKFold(labels,n_folds=10, random_state=1000,shuffle=True)
for train_index, test_index in sss:
    break

train_x, train_y = X_train.values[train_index], labels.values[train_index]
test_x, test_y = X_train.values[test_index], labels.values[test_index]

### building the classifiers
clfs = []
reg1=ensemble.GradientBoostingRegressor(n_estimators=800,learning_rate=0.03,max_features=0.7,max_depth=5)#0.41503
reg2=ensemble.RandomForestRegressor(n_estimators=600,min_samples_split=6,max_features=0.8)   #0.42462 
reg3=ensemble.ExtraTreesRegressor(n_estimators=1100,min_samples_split=10,max_features=1.0)   #0.40683

reg1.fit(train_x, np.log(train_y+1))
print('REG1 rmsle {score}'.format(score=mean_squared_error(np.log(test_y+1), reg1.predict(test_x))** 0.5))
clfs.append(reg1)

### usually you'd use xgboost and neural nets here

reg2.fit(train_x, np.log(train_y+1))
print('REG2 rmsle {score}'.format(score=mean_squared_error(np.log(test_y+1), reg2.predict(test_x))** 0.5))
clfs.append(reg2)

reg3.fit(train_x, np.log(train_y+1))
print('REG3 rmsle {score}'.format(score=mean_squared_error(np.log(test_y+1), reg3.predict(test_x))** 0.5))
clfs.append(reg3)

### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(np.exp(clf.predict(test_x))-1)

def rmsle_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    #we need to normalize them first
    weights/=np.sum(weights)
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        #if final_prediction is None:
        #    final_prediction = weight*prediction
        #else:
            final_prediction += weight*prediction

    return calculate_rmsle(final_prediction,test_y)
    
#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(rmsle_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

#REG1 rmsle 0.264844810765
#REG2 rmsle 0.282573244073
#REG3 rmsle 0.282085384757
#Ensamble Score: 0.261384339451
#Best Weights: [  7.25553216e-01   9.97465999e-18   2.74446784e-01]
