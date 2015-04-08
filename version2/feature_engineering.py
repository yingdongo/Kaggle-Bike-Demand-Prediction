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
    data['temp_diff']=abs(data.atemp-data.temp)
    data['season1']=data.season==1
    data['season2']=data.season==2
    data['season3']=data.season==3
    data['season4']=data.season==4
    data['weather1']=data.weather==1
    data['weather2']=data.weather==2
    data['weather3']=data.weather==3
    data['weather4']=data.weather==4    
    transform_datetime(data)
    data['sunday']=data.weekday==6
    data['month1']=data.month==1
    data['month2']=data.month==2
    data['month3']=data.month==3
    data['month4']=data.month==4
    data['month5']=data.month==5
    data['month6']=data.month==6
    data['month7']=data.month==7
    data['month8']=data.month==8
    data['month9']=data.month==9
    data['month10']=data.month==10
    data['month11']=data.month==11
    data['month12']=data.month==12
def main():
    train=load_data('train.csv')
    add_feature(train)

#if __name__ == '__main__':
    #main()
