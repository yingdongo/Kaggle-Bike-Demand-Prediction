# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:03:58 2015

@author: Ying
"""
import time
def data_preprocess(data):
     i = 0
     for timestamp in data['datetime']:
         i += 1
         date_object = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
         data.loc[i-1, 'year'] = date_object.tm_year-2011
         data.loc[i-1, 'weekday'] = date_object.tm_wday
         #data.loc[i-1,'month']=date_object.tm_mon
         data.loc[i-1, 'hour'] = date_object.tm_hour
         data.loc[i-1, 'day'] = date_object.tm_mday
     return data