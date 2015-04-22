# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:55:00 2015

@author: Ying
"""

from matplotlib import pyplot as plt
import numpy as np
from tools import load_data

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

if __name__ == '__main__':
    main()
