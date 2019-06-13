#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:43:41 2019

@author: elizabethhutton
"""

import urllib
import json
import pandas as pd

#PROTOTYPE 

#starting with only AK cases from 2008
api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'
url = urllib.request.urlopen(api_AK)
obj = json.load(url)

data = obj['results']
case1 = data[1]

#fields = ['casebody','court','decision_date','docket_number','id','name','name_abbreviation']

#extract nonduplicate opinions from each case 
docs = set()
count = 0
for d in data: 
    if 'casebody' in d: 
        body = d['casebody']
        if 'data' in body: 
            data = body['data']
            if 'opinions' in data: 
                ops = data['opinions']
                l = list()
                for op in ops: 
                    if 'text' in op: 
                        docs.add((count,op['text']))
                    else: print("No 'text' key in ops.")
            else: print("No 'opinions' key.")
            count+=1
        else: print("No 'data' key.")
    else: print("No 'casebody' key.")
    
#convert to dataframe and save
df = pd.DataFrame(list(docs))
df.columns = ['case','text']
df['text'] = df['text'].astype(str)
filename = 'documents_raw.pkl'
df.to_pickle(filename)


            
                
                
        
    
    