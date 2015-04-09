# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:40:28 2015

@author: Ying
"""
import pandas as pd

def load_data(url):
    data = pd.read_csv(url)
    return data

