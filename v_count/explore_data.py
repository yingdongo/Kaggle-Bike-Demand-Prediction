# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:55:00 2015

@author: Ying
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('ggplot')
from data_preprocess import data_preprocess
from pandas.tools.plotting import scatter_matrix
def load_data(url):
    data = pd.read_csv(url)
    return data

def show_data(data):
    print data.head(10)
    #have a look at few top rows
    print data.describe()
    #describe() function would provide count, mean, standard deviation (std), min, quartiles and max in its output

def plot_box(data):
    plt.figure()
    plt.boxplot(data)
    plt.show()
    
def plot_correlations(data):
    """Plot pairwise correlations of features in the given dataset"""

    from matplotlib import cm
    
    cols = data.columns.tolist()
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    # Plot absolute value of pairwise correlations since we don't
    # particularly care about the direction of the relationship,
    # just the strength of it
    cax = ax.matshow(data.corr().abs(), cmap=cm.YlOrRd)
    
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(cols)
def main(): 
    train=load_data('train.csv')
    data_preprocess(train)
    
    show_data(train)
    cols1=['season','holiday', 'workingday', 'weather']  
    cols2=['temp','atemp','humidity','windspeed']
    cols3=['casual','registered','count']
    plt.figure()
    train[cols1].boxplot()
    plt.show()
    plt.figure()
    train[cols2].boxplot()
    plt.show()
    plt.figure()
    train[cols3].boxplot()
    plt.show()
    plot_correlations(train.iloc[:,1:12])

#if __name__ == '__main__':
#    main()

train=load_data('train.csv')
#train=train.drop(['datetime'],axis=1)
data_preprocess(train)
def data_plot():
    countw=train[['count','dayofweek']]
    cols =np.array(['Sun','Mon','Tue','Wed','Thu','Fr','Sat'])
    countw.boxplot(by='dayofweek',figsize=(10, 10))
    
    countw=train[['count','hour']]
    week_count = countw.groupby('hour')
    count_mean=week_count.mean()
    cols=range(0,24)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.plot(count_mean)
    plt.ylabel('[count]')
    plt.xlabel('[hour]')
    plt.title('mean value of count per hour')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    
    countr=train[['registered','dayofweek']]
    countr.boxplot(by='dayofweek',figsize=(10, 5))
    
    countc=train[['casual','dayofweek']]
    countc.boxplot(by='dayofweek',figsize=(10, 5))

def plot_hour_mean():
    count=train[['count','hour']]
    g_count = count.groupby('hour')
    hour_count=g_count.mean()
    registered=train[['registered','hour']]
    g_registered = registered.groupby('hour')
    hour_registered=g_registered.mean()
    casual=train[['casual','hour']]
    g_casual = casual.groupby('hour')
    hour_casual=g_casual.mean()
    cols=range(0,24)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.plot(hour_count)
    plt.plot(hour_registered)
    plt.plot(hour_casual)
    plt.legend(['count', 'registered', 'casual'], loc='upper left')
    plt.xlabel('[hour]')
    plt.ylabel('Number of users')

    plt.title('mean value of count/registered/casual per hour')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)

def plot_day_mean():
    count=train[['count','dayofweek']]
    g_count = count.groupby('dayofweek')
    hour_count=g_count.mean()
    registered=train[['registered','dayofweek']]
    g_registered = registered.groupby('dayofweek')
    hour_registered=g_registered.mean()
    casual=train[['casual','dayofweek']]
    g_casual = casual.groupby('dayofweek')
    hour_casual=g_casual.mean()
    cols=range(1,8)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    plt.plot(hour_count,linewidth=2)
    plt.plot(hour_registered,linewidth=2)
    plt.plot(hour_casual,linewidth=2)
    plt.legend(['count', 'registered', 'casual'],bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.xlabel('[dayofweek]')
    plt.ylabel('Number of users')

    plt.title('mean value of count/registered/casual per dayofweek')
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fr','Sat','Sun'])
    ax.set_xticks(np.arange(len(cols)))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    

plot_day_mean()
plot_hour_mean()