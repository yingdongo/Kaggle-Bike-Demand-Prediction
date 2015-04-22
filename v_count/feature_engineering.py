# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:03:22 2015

@author: Ying
"""
def feature_engineering(data):
    data['temp_diff']=abs(data.atemp-data.temp)
    return data
    