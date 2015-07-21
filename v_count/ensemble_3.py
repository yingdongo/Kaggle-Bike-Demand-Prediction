# -*- coding: utf-8 -*-
"""
Created on Mon May 25 01:16:05 2015

@author: Ying
"""
import pandas as pd
import numpy as np
pred1=pd.read_csv('to_submit/log_count_param_solution_gbr800.3.75_month.csv')
pred2=pd.read_csv('to_submit/log_count_param_solution_rf60060.8_month.csv')
pred3=pd.read_csv('submissions/log_count_param_solution_ex_month.csv')

pred=0.725553216*pred1['count']+9.97465999e-18*pred2['count']+0.274446784*pred3['count']
#pred=(pred1['count']+pred3['count']+pred2['count'])/3

def write_submission(filename,test_ids,preds):
    with open(filename, "wb") as outfile:
         outfile.write("datetime,count\n")
         for e, val in enumerate(preds):
             outfile.write("%s,%s\n"%(test_ids[e],np.around(abs(val))))

write_submission('to_submit/ensemble_gbr_rf_ex_linear.csv',pred1['datetime'],list(pred))

#0.40712 average 0.40816 0.37175