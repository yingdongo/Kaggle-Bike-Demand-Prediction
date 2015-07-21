# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:52:14 2015

@author: Ying
"""


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedKFold
from tools import Reg
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
y=train['count']
labels = train[['registered','casual']]

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
sss=StratifiedKFold(y,n_folds=10, random_state=1000,shuffle=True)
for train_index, test_index in sss:
    break
train_x, train_y = X_train.values[train_index], labels.values[train_index]
test_x, test_y = X_train.values[test_index], y.values[test_index]

train_x=pd.DataFrame(data=train_x,columns=feature_cols)
train_y=pd.DataFrame(data=train_y,columns=['registered','casual'])
test_x=pd.DataFrame(data=test_x,columns=feature_cols)

### building the classifiers
clfs = []

#[0.30175291564177587, 'gbr', 'gbr', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}]
r10=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)#0.302298687188
r11=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5) #0.36899
reg1= Reg(r10,r11)
#==============================================================================
# 
# #[0.31698795457988432, 'extra trees', 'extra trees', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 10, 'n_estimators': 1000}]
# r20=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)    
# r21=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=10)    #0.317153927279
# reg2=Reg(r20,r21)
# 
# #[0.32037254768863044, 'random forest', 'random forest', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 8, 'n_estimators': 500}]
# r30=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
# r31=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=8) #0.321115940898
# reg3=Reg(r30,r31)
#==============================================================================

r40=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)
r41=ensemble.ExtraTreesRegressor(n_estimators= 1000,min_samples_split=10) #0.302394242383   
reg4=Reg(r40,r41)

#[0.29793958119927855, 'gbr', 'random forest', {'n_estimators': 700, 'learning_rate': 0.03, 'max_depth': 5}, {'min_samples_split': 8, 'n_estimators': 500}]
r50=ensemble.GradientBoostingRegressor(n_estimators= 700,learning_rate=0.03,max_depth=5)
r51=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=8)   #0.302564934886 
reg5=Reg(r50,r51)

#[0.3066113932919306, 'extra trees', 'gbr', {'min_samples_split': 10, 'n_estimators': 900}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
#==============================================================================
r60=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)    #0.313161123986
r61=ensemble.GradientBoostingRegressor(n_estimators= 800,learning_rate=0.03,max_depth=3)
reg6=Reg(r60,r61)
# 
# #[0.31052237683458117, 'extra trees', 'random forest', {'min_samples_split': 10, 'n_estimators': 900}, {'min_samples_split': 6, 'n_estimators': 500}]
# r70=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)#0.316701989476    
# r71=ensemble.RandomForestRegressor(n_estimators= 500,min_samples_split=6)    
# reg7=Reg(r70,r71)   
# 
# #[0.30727972714289437, 'random forest', 'gbr', {'min_samples_split': 8, 'n_estimators': 600}, {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 3}]
# r80=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
# r81=ensemble.GradientBoostingRegressor(n_estimators= 800,learning_rate=0.03,max_depth=8) #
# reg8=Reg(r80,r81)
# 
# #[0.31090319098589592, 'random forest', 'extra trees', {'min_samples_split': 8, 'n_estimators': 600}, {'min_samples_split': 10, 'n_estimators': 900}]
# r90=ensemble.RandomForestRegressor(n_estimators= 600,min_samples_split=8)    
# r91=ensemble.ExtraTreesRegressor(n_estimators= 900,min_samples_split=10)#0.321018320502
#==============================================================================
#reg9=Reg(r90,r91)

reg1.fit(train_x, train_y)
print('REG1 rmsle {score}'.format(score=calculate_rmsle(reg1.predict(test_x),test_y)))
clfs.append(reg1)

#==============================================================================
# reg2.fit(train_x, train_y)
# print('REG2 rmsle {score}'.format(score=calculate_rmsle(reg2.predict(test_x),test_y)))
# clfs.append(reg2)
# 
# reg3.fit(train_x, train_y)
# print('REG3 rmsle {score}'.format(score=calculate_rmsle(reg3.predict(test_x),test_y)))
# clfs.append(reg3)
#==============================================================================

reg4.fit(train_x, train_y)
print('REG4 rmsle {score}'.format(score=calculate_rmsle(reg4.predict(test_x),test_y)))
clfs.append(reg4)

reg5.fit(train_x, train_y)
print('REG5 rmsle {score}'.format(score=calculate_rmsle(reg5.predict(test_x),test_y)))
clfs.append(reg5)

#==============================================================================
reg6.fit(train_x, train_y)
print('REG6 rmsle {score}'.format(score=calculate_rmsle(reg6.predict(test_x),test_y)))
clfs.append(reg6)
# 
# reg7.fit(train_x, train_y)
# print('REG7 rmsle {score}'.format(score=calculate_rmsle(reg7.predict(test_x),test_y)))
# clfs.append(reg7)
# 
# reg8.fit(train_x, train_y)
# print('REG8 rmsle {score}'.format(score=calculate_rmsle(reg8.predict(test_x),test_y)))
# clfs.append(reg8)
# 
# reg9.fit(train_x, train_y)
# print('REG9 rmsle {score}'.format(score=calculate_rmsle(reg9.predict(test_x),test_y)))
#==============================================================================
#clfs.append(reg9)

### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(clf.predict(test_x))

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

#REG1 rmsle 0.271876025508
#REG2 rmsle 0.280959444996
#REG3 rmsle 0.283898497181
#REG4 rmsle 0.269957419786
#REG5 rmsle 0.269873551814
#REG6 rmsle 0.281544032235
#REG7 rmsle 0.27921747435
#REG8 rmsle 0.280453324929
#REG9 rmsle 0.284520873832
#Ensamble Score: 0.264809965874
#Best Weights: [  2.33866329e-01   7.09082595e-02   2.24565375e-17   1.79635279e-01
#   2.04723916e-01  -1.25767452e-17   1.10315011e-01   2.00551206e-01
#   5.25838054e-18]