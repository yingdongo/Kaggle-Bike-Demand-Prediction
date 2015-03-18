# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:03:22 2015

@author: Ying
"""
import time
from explore_data import load_data
def transform_datetime(data):
     i = 0
     for timestamp in data['datetime']:
         i += 1
         date_object = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

         data.loc[i-1, 'weekday'] = date_object.tm_wday
         data.loc[i-1,'month']=date_object.tm_mon
         data.loc[i-1, 'hour'] = date_object.tm_hour

def add_feature(data):
    data['temp_diff']=data.atemp-data.temp
    transform_datetime(data)
def main():
    train=load_data('train.csv')
    add_feature(train)

#if __name__ == '__main__':
    #main()
