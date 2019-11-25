# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:09:35 2018

@author: ddm
"""
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize


file1 = '_bls_session_Ddm1_2019-10-21-18-04-28.json'
file2 = '2019-10-21-16-07-20_CPRTutor_annotations.json'

if file1:
    with open(file1, 'r') as f:
        data = json.load(f)
    df1 = pd.DataFrame()
    df1 = pd.concat([pd.DataFrame(data),
                    json_normalize(data['intervals'])],
                   axis=1).drop('intervals', 1)
    df1 = df1.apply(pd.to_numeric, errors='ignore')
    df1.columns = df1.columns.str.replace("annotations.", "")
    offset = pd.to_timedelta('00:00:10.850')
    df1.start = pd.to_timedelta(df1.start) + offset
    df1.end = pd.to_timedelta(df1.end) + offset
    #print(df1[['start','end']])


if file2:
    with open(file2, 'r') as f:
        data = json.load(f)
    df2 = pd.DataFrame()
    df2 = pd.concat([pd.DataFrame(data),
                    json_normalize(data['Intervals'])],
                   axis=1).drop('Intervals', 1)
    df2 = df2.apply(pd.to_numeric, errors='ignore')
    df2.columns = df2.columns.str.replace("annotations.", "")
    df2.start = pd.to_timedelta(df2.start)
    df2.end = pd.to_timedelta(df2.end)
    #print(df2[['start','end']])

s = df2.start-df1.start<pd.to_timedelta('00:00:01.000')
while s.any():
    df1 = df1.drop(s[s == True].index[0])
    df1 = df1.reset_index(drop=True)
    print("deleting "+str(s[s == False].index[0]))
    print (df2.start - df1.start)
    s = df2.start - df1.start < pd.to_timedelta('00:00:01.000')

print(df1.reset_index(drop=True))